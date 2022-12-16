import random

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

repo = "runwayml/stable-diffusion-v1-5"
euler_ancestral_scheduler = EulerAncestralDiscreteScheduler.from_config(repo, subfolder="scheduler")

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None}, allow_output_mutation=True)
def load_model():
    return StableDiffusionPipeline.from_pretrained(
        repo, scheduler=euler_ancestral_scheduler, torch_dtype=torch.float32, revision="fp16"
    ).to("cpu")
#pipe.enable_sequential_cpu_offload()
#pipe.enable_attention_slicing(1)

diffuser = load_model()

# 웹 페이지 제목
st.title("그림을 그려드려요! 🤓")
# 웹 페이지 부제목
st.subheader('몇 가지 제시어를 부탁드릴게요!')

# 웹 페이지에 입력
with st.form(key="form"):
    prompt = st.text_input(label="제시어", placeholder="화성에서 말을 타고 있는 우주인 사진")
    submit = st.form_submit_button("Go!")

# 버튼을 눌렀을 때, 작동되는 코드
if submit:
    st.write("그리는 중이에요!... 🤓")

    # 모델의 inference가 끝날 때까지 기다림
    with st.spinner("그리는 중이에요... 🤓"):
        # classifier 라는 이름의 딥러닝 모델사용.
        # 원하는 모델로 변경할 수 있다.
        seed = random.randrange(10000, 99999)
        generator = torch.Generator().manual_seed(seed)
        print(prompt)
        results = diffuser(prompt,num_inference_steps=25, generator=generator).images[0]

    st.image(results, caption=prompt)