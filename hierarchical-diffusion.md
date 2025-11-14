- 1. 遍历每个prompt，调用KandinskyV22PriorPipeline模型得到其有条件的正向嵌入和无条件的负向嵌入，并堆叠起来（[2, prompt数量, 嵌入维度]）
```py
for prompt in tqdm(prompts):
    image_embeddings.append(compute_prior(md, prompt))
del md.prior_pipe

_ = md.prior_pipe(prompt).to_tuple()[0]
```
- 2. 提取所有正向嵌入的向量集合，并计算他们两两之间的余弦距离矩阵
```py
distance_matrix = pdist(image_embeddings[0].cpu().numpy(), metric="cosine")
```
- 3. 调用linkage函数进行hierarchical clustering，使用ward聚类方法，返回一个linkage matrix，每一行表示一次聚类操作，包含被合并的两个簇的索引、合并后的距离、以及新簇的样本数
```py
Z = sch.linkage(distance_matrix, method='ward', metric='cosine')
```
举例：
假设有4个prompt，嵌入维度为128，则
输入shape: (4, 128)
距离矩阵shape: (6,)，分别是(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)这6对的距离
连接矩阵shape: (3, 4)，表示3次聚类操作，eg: 
```py
array([
    [0, 1, 0.12, 2],  # 第一次合并，点0和点1，距离0.12，合并后有2个点
    [2, 3, 0.20, 2],  # 第二次合并，点2和点3，距离0.20，合并后有2个点
    [4, 5, 0.35, 4]   # 第三次合并，前两个新簇（索引4和5），距离0.35，合并后有4个点
    ])
```
- 4. 计算threshold intervals
生成从0到cfg.tau的等间距数组，一共cfg.inference_steps个点，从大到小排列，用于控制聚类的分裂或合并
```py
phi = np.linspace(0, cfg.tau, cfg.inference_steps)[::-1]
```
- 5. 调用AutoPipelineForText2Image.decoder_pipe.prepare_latents()以随机噪声初始化latents，形状为(1, num_channels_latents, height, width)
```py
x_k = md.pipe.decoder_pipe.prepare_latents(
    (1, num_channels_latents, height, width),
    image_embeddings[0].dtype,
    md.device,
    None,
    None,
    md.scheduler,
)
```
- 6. 初始化上一轮聚类的标签，每个prompt初始都属于自己的独立簇（初始化为1）
```py
C_prev = np.ones(num_prompts, dtype=int)
```
- 7. 开始循环去噪
    - 8. 对于当前步的threshold phi_k，调用fcluster根据linkage matrix Z对prompt进行聚类，得到当前步的标签labels_k
    ```py
    labels_k = sch.fcluster(Z, phi_k, criterion='distance')
    ```
    举例：
    ```py
    array([
        [0, 1, 0.12, 2],  # 第一次合并，点0和点1，距离0.12
        [2, 3, 0.20, 2],  # 第二次合并，点2和点3，距离0.20
        [4, 5, 0.35, 4]   # 第三次合并，前两个新簇，距离0.35
    ])
    ```
    phi_k = 0.1时，阈值比所有合并距离都小，则每个点都属于自己的簇，labels_k = [1, 2, 3, 4]；phi_k = 0.15时，阈值大于第一次合并（0.12），小于第二次（0.20），点0和点1被合并为一类，点2为一类，点3为一类，labels_k = [1, 1, 2, 3]；phi_k = 0.25时，阈值大于前两次合并（0.12, 0.20），小于最后一次（0.35），点0和点1一类，点2和点3一类，labels_k = [1, 1, 2, 2]；phi_k = 0.4，阈值大于所有合并距离，所有点都被合并为一类，labels_k = [1, 1, 1, 1]
    - 9. C_k为当前步所有簇的聚类标签号，first_index为每个聚类标签第一次出现的位置索引，inv为每个prompt属于哪个聚类标签的逆索引，counts为每个聚类标签下成员的数量
    ```py
    C_k, first_index, inv, counts = np.unique(labels_k, return_inverse=True, return_index=True, return_counts=True)
    ```
    - 10. 取出每个当前簇的第一个成员在上一轮属于哪个簇，建立当前簇到父簇的映射parent_map，追踪每个新簇是由上一轮的哪个父簇分裂出来的，然后更新C_prev
    ```py
    parent_map = dict(zip(C_k, C_prev[first_index]))
    ```
    - 11. 用父簇索引，从上一轮的latent x_k中复制出每个当前簇的初始潜变量latent
    ```py
    C_parent = np.array([parent_map[c] for c in C_k]) - 1
    x_k = x_k[C_parent]
    ```
    - 12. 对每个聚类，计算其成员的嵌入均值，得到每个簇的代表性嵌入
    ```py
    y_hat = compute_mean_embeddings(
        embeddings=image_embeddings,
        inv=inv_t,
        counts=counts_t
    )
    ```
    - 13. 对每个聚类标签c_idx，记录所有属于该簇的prompt索引
    ```py
    all_prompt_indices = []
    for c_idx in C_k:
        prompt_indices = np.where(inv == c_idx-1)[0]
        all_prompt_indices.append(prompt_indices)
    ```
    - 14. 把当前聚类的均值嵌入作为条件，传递给扩散模型的去噪过程
    ```py
    added_cond_kwargs = {"image_embeds": current_image_embeddings}
    ```
    - 15. 收集每个聚类簇下所有的prompts，并对每类的prompt用逗号拼接成一个字符串
    ```py
    for group in all_prompt_indices:
        prompt_clusters.append(prompt_tracker[group])
        prompt_cluster_flat.append(", ".join(prompt_tracker[group]))
    ```
    - 16. 去噪
    ```py
    x_k = md.denoise(
        x_k,
        text_embeddings=None,
        t=timesteps[k],
        added_cond_kwargs=added_cond_kwargs,
    )
    ```
- 17. 将latents解码为图像
```py
denoised_images = md.decode_latents(x_k)
```






