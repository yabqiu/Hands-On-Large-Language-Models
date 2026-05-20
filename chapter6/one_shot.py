import kick_zscaler

from transformers import pipeline, GenerationConfig

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen3.5-4B",
    device_map="mps",
    return_full_text=True,
)

zeroshot_tot_prompt = [
    {"role": "user", "content":
        ("Imagine three different experts are answering, "
         "this question. All experts will write down 1 step of their thinking, then share "
         "it with the group. Then all experts will go on to the next step, etc. If any "
         "expert realizes they're wrong at any point then they leave. The question is "
         "'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, "
         "how many apples do they have?' Make sure to discuss the results.")
     }
]

output = pipe(zeroshot_tot_prompt, generation_config=GenerationConfig(do_sample=False, max_new_tokens=50000, max_length=None))
print(output[0]["generated_text"])
