<!-- 标题 -->
<h1 align="center">ChatYuan-7B-merge</h1>

<!-- 图标 -->
<p align="center">
<!--   <a href="https://www.cnblogs.com/tiansz/p/17318568.html">
    捐赠
  </a>&nbsp; &nbsp;  -->
  <a href="https://space.bilibili.com/28606893?spm_id_from=333.1007.0.0">
    bilibili
  </a>&nbsp; &nbsp; 
  <a href="https://www.cnblogs.com/tiansz/">
    blog
  </a>&nbsp; &nbsp;
<!--   <a href="https://www.douyin.com/user/MS4wLjABAAAAqkpp6UyrANDXFStAMWuRPp7FU4zHfyq0_OYPoC75_qQ">
    抖音
  </a>&nbsp; &nbsp; -->
  <a href="https://www.kaggle.com/tiansztianszs">
    kaggle
  </a>&nbsp; &nbsp;
  <a href="https://huggingface.co/tiansz">
    huggingface
  </a>
</p>

<!-- 项目介绍 -->
<p align="center">Based on LLAMA's latest Chinese-English dialogue language large model</p>

<br>

You can see more detail in this [repo](https://github.com/clue-ai/ChatYuan-7B)

<br>

## How to use
```python
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

ckpt = "tiansz/ChatYuan-7B-merge"
device = torch.device('cuda')
model = LlamaForCausalLM.from_pretrained(ckpt)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

def answer(prompt):
  prompt = f"用户：{prompt}\n小元："
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
  generate_ids = model.generate(input_ids, max_new_tokens=1024, do_sample = True, temperature = 0.7)
  output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  response = output[len(prompt):]
  return response

result = answer("你好")
print(result)
```

<br>
int8:

```python
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

ckpt = "tiansz/ChatYuan-7B-merge"
device = torch.device('cuda')
max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-1}GB'
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
model = LlamaForCausalLM.from_pretrained(ckpt, device_map='auto', load_in_8bit=True, max_memory=max_memory)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

def answer(prompt):
  prompt = f"用户：{prompt}\n小元："
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
  generate_ids = model.generate(input_ids, max_new_tokens=1024, do_sample = True, temperature = 0.7)
  output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  response = output[len(prompt):]
  return response

result = answer("你好")
print(result)
```

<br>

## License
- [ChatYuan-7B](https://github.com/clue-ai/ChatYuan-7B)
- [llama](https://github.com/facebookresearch/llama)
