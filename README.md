# diffusers_liuman
1. SD-HM0.4.0 fully finetune
  训练好的模型：
  /mnt/share_disk/liuman/model/SD-HMft0.8.0_12withtime
  /mnt/share_disk/liuman/model/SD-HMft0.8.1_24withtime
  /mnt/share_disk/liuman/model/SD-HMft0.8.2_withtime
  数据集：
    图片：/mnt/share_disk/liuman/data/blip_tgt/imgs
    prompt不加时间描述：/mnt/share_disk/liuman/data/blip_tgt/pmps
    加12小时制时间描述：/mnt/share_disk/liuman/data/blip_tgt_twelve/pmps
    
  代码基于https://github.com/Beaconsyh08/diffusers/tree/syh 修改
   1.1 在SD-HM0.4.0模型的unet配置文件中添加/修改参数：
      "addition_embed_type": "text_time",
      "addition_time_embed_dim": 2816,
      "projection_class_embeddings_input_dim": 2816,

   1.2 修改diffusers/examples/text_to_image/train_text_to_image.py
       a. 修改preprocess_train方法，获取examples["image_file_names"]，image_file_names是对图片的prompt，包含了对图片的时间描述，之后需            要根据examples["image_file_names"]提取时间信息。
           def preprocess_train(examples):
                # 图像被转换为张量并存储在 examples 字典中的 "pixel_values" 键中
                images = [PIL.Image.open(image).convert("RGB") for image in examples[image_column]]
                examples["pixel_values"] = [train_transforms(image) for image in images]
                examples["input_ids"] = tokenize_captions(examples)
                
                # 添加文件名信息
                examples["image_file_names"] = [os.path.basename(image_path) for image_path in examples[image_column]]
                return examples

       b. 修改collate_fn方法，提取时间信息image_times（0-23的整数）
           # 1.从文件名中提取时间信息的函数
            def extract_time_from_filename(filename):
                # parts = filename.split(' ')
                parts = re.split(r'[ \-]', filename)
                hour_str = parts[2]
                try:
                    hour = int(hour_str)
                    return hour
                except ValueError:
                    raise ValueError(f"Unable to extract hour from filename: {filename}")
        
            def collate_fn(examples):
                pixel_values = torch.stack([example["pixel_values"] for example in examples])
                pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
                input_ids = torch.stack([example["input_ids"] for example in examples])
        
                # 2.提取时间信息
                image_file_names = [example["image_file_names"] for example in examples]
                image_times = [extract_time_from_filename(filename) for filename in image_file_names]
                batch = {"pixel_values": pixel_values, "input_ids": input_ids, "image_times": image_times}
                return batch

       c. 迭代时：
         for epoch in range(first_epoch, args.num_train_epochs):
             unet.train()
             train_loss = 0.0
             for step, batch in enumerate(train_dataloader):
               # 获取图片时间信息
                image_times = batch["image_times"]
                image_times = torch.tensor(image_times)
                image_times = image_times.to(accelerator.device, dtype=weight_dtype)
         
               # 把condition加入unet
                unet_added_conditions = {"time_ids": image_times}
                model_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_conditions
                    ).sample

   1.3 修改diffusers/src/diffusers/models/unet_2d_condition.py
     a. 在合适的位置添加
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
   
        elif self.config.addition_embed_type == "text_time":
                # SDXL - style
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")    # torch.Size([4])
                time_embeds = self.add_time_proj(time_ids.flatten())    # torch.Size([4, 2816])
                add_embeds = time_embeds.to(emb.dtype)  # torch.Size([4, 2816])
                device = emb.device
                add_embeds = add_embeds.to(device)
                aug_emb = self.add_embedding(add_embeds)
                aug_emb = aug_emb.to(device)
                emb = emb + aug_emb     # emb:torch.Size([4, 1280])

   1.4 推理
   修改/mnt/ve_share/liuman/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_liu.py
   a. 添加
        # 使用逗号分割字符串
        time_and_description = prompt.split(' ')
        # 获取时间部分
        time = time_and_description[4]
        # 提取小时部分
        image_times = int(time.split(':')[0])
        image_times = torch.tensor([image_times])

    b. 迭代时添加：
      for i, t in enumerate(timesteps):
          added_cond_kwargs = {"time_ids": image_times}
          noise_pred = self.unet(
              latent_model_input,
              t,
              encoder_hidden_states=prompt_embeds,
              cross_attention_kwargs=cross_attention_kwargs,
              added_cond_kwargs=added_cond_kwargs,
              return_dict=False,
          )[0]
    

3. instruct-pix2pix fully finetune
   训练好的模型：
   /mnt/ve_share/liuman/model/instructPix2Pix_0.6
   /mnt/ve_share/liuman/model/instructPix2Pix_0.8
   数据集：
    /mnt/share_disk/liuman/datainstrPix2Pix/0.6
    /mnt/share_disk/liuman/datainstrPix2Pix/0.8
   
   代码基于https://github.com/Beaconsyh08/diffusers/tree/syh 修改
   2.1 在base_instructpix2pix模型（/mnt/ve_share/liuman/model/instruct-pix2pix）模型的unet配置文件中添加/修改参数：
    "addition_embed_type": "text_time",
    "addition_embed_type_num_heads": 64,
    "addition_time_embed_dim": 2816,
    "projection_class_embeddings_input_dim": 2816,

   2.2 修改diffusers/examples/instruct_pix2pix/train_instruct_pix2pix.py
     a. 添加/修改
        # 新的函数用于提取时间并转换为24小时制的整数
      def extract_time(caption):
          # 使用正则表达式匹配时间格式，假设时间格式为"6:00pm"这种形式
          match = re.search(r'(\d{1,2}:\d{2}(am|pm))', caption)
          if match:
              time_str = match.group(0)
              # 解析时间
              hour, minute = map(int, re.findall(r'\d+', time_str))
              if "pm" in time_str.lower() and hour < 12:
                  hour += 12
              return hour
          return None  # 如果未找到时间，则返回None
  
      def preprocess_train(examples):
          # Preprocess images.
          preprocessed_images = preprocess_images(examples)
          # Since the original and edited images were concatenated before
          # applying the transformations, we need to separate them and reshape
          # them accordingly.
          original_images, edited_images = preprocessed_images.chunk(2)
          original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
          edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)
  
          # Collate the preprocessed images into the `examples`.
          examples["original_pixel_values"] = original_images
          examples["edited_pixel_values"] = edited_images
  
          # Preprocess the captions.  captions就是edit_prompt   ['make it 6:00am']
          captions = list(examples[edit_prompt_column])
          image_times = [extract_time(caption) for caption in captions]   # [6]
          # 将image_times转化成张量
          image_times = torch.tensor(image_times, dtype=torch.int)
          examples["input_ids"] = tokenize_captions(captions)   # torch.Size([1, 77])
          examples["image_times"] = image_times
          return examples

       def collate_fn(examples):
          original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
          original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
          edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
          edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
          input_ids = torch.stack([example["input_ids"] for example in examples])
          image_times = torch.stack([example["image_times"] for example in examples])
          return {
              "original_pixel_values": original_pixel_values,
              "edited_pixel_values": edited_pixel_values,
              "input_ids": input_ids,
              "image_times": image_times,
          }

     b. 迭代时：
       for epoch in range(first_epoch, args.num_train_epochs):
          unet.train()
          train_loss = 0.0
          for step, batch in enumerate(train_dataloader):

             image_times = batch["image_times"]
             image_times = torch.tensor(image_times)   
             image_times = image_times.to(accelerator.device, dtype=weight_dtype)

             unet_added_conditions = {"time_ids": image_times}
             model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_conditions).sample

   2.3 推理脚本
   （/mnt/ve_share/liuman/inference/inf_instructPix2Pix.py）
   /mnt/ve_share/liuman/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
     a. 添加
         # 获取时间对应的24小时数
        time_str = prompt.split(" ")[-1]
        time_obj = datetime.strptime(time_str, "%I:%M%p")
        image_times = time_obj.hour
        image_times = torch.tensor([image_times])
        image_times = image_times.to(device).repeat(batch_size * num_images_per_prompt, 1)
     b. 迭代时添加：
        for i, t in enumerate(timesteps):
           added_cond_kwargs = {"time_ids": image_times}
           noise_pred = self.unet(
               scaled_latent_model_input, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs, return_dict=False
                )[0]
   

   
   
