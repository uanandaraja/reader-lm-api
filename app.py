from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import requests
from vllm import LLM, SamplingParams

app = FastAPI()

# Model configuration
MODEL_NAME = 'jinaai/reader-lm-1.5b'
SAMPLING_PARAMS = SamplingParams(
    temperature=0,
    top_k=1,
    presence_penalty=0.25,
    repetition_penalty=1.08,
    max_tokens=1024
)

# Initialize LLM
llm = LLM(model=MODEL_NAME, dtype='float16', gpu_memory_utilization=0.95)


class URLInput(BaseModel):
    url: str


def clean_html(html: str):
    patterns = [
        r'<[ ]*script.*?\/[ ]*script[ ]*>',
        r'<[ ]*style.*?\/[ ]*style[ ]*>',
        r'<[ ]*meta.*?>',
        r'<[ ]*!--.*?--[ ]*>',
        r'<[ ]*link.*?>'
    ]
    for pattern in patterns:
        html = re.sub(pattern, '', html, flags=(
            re.IGNORECASE | re.MULTILINE | re.DOTALL))
    return html


def get_html_content(url: str) -> str:
    api_url = f'https://r.jina.ai/{url}'
    headers = {'X-Return-Format': 'html'}
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/convert")
async def convert_to_markdown(input: URLInput):
    # Get HTML
    html = get_html_content(input.url)

    # Clean HTML
    html = clean_html(html)

    # Create prompt and generate
    prompt = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": html}],
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate markdown
    results = llm.generate(prompt, sampling_params=SAMPLING_PARAMS)

    return {"markdown": results[0].outputs[0].text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
