from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Optional,List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import nltk
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer
from starlette.concurrency import run_in_threadpool
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import traceback


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
tokenizer = RegexpTokenizer(r"\w+")
lmtzr = WordNetLemmatizer()
# client_async = AsyncOpenAI(api_key="key")


def get_nouns(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [lmtzr.lemmatize(word) for word in tokens if word not in stop_words]
    pos_tags = nltk.pos_tag(tokens)
    nouns = [word for word, tag in pos_tags if tag.startswith("NN")]
    noun_freq = Counter(nouns)
    top_nouns = [word for word, _ in noun_freq.most_common(3)]
    return top_nouns


class TextInput(BaseModel):
    text: str
    
class OuputSchema(BaseModel):
    title: Optional[str]=Field(None,description="Title if present in the text")
    topics: List[str]=Field(...,description="3 key topics in the text")
    sentiment: str=Field(...,description="Title if present in the text")
    summary : str=Field(...,description="Summary of the text")

app = FastAPI()


@app.post("/analyze")
async def get_summary(input: TextInput):
    """get summary along with tile, key topics and keywords"""
    try:
        data = input.text
        completion =  await client_async.beta.chat.completions.parse(
                            model="gpt-4.1-nano",
                        messages=[
                            {"role": "user", "content": f"""Extract title, 3 key topics ,the overall sentiment and generate 1-2 line summary of the given text. If title is not present in the text return None. text: {data} """}
                        ],
                        response_format=OuputSchema,
                        temperature= 0.1
                    )
        result = completion.choices[0].message.parsed
        keywords = await run_in_threadpool(get_nouns,data)
        return JSONResponse({"summary":result.summary,"title": result.title, "topics": result.topics,"sentiment":result.sentiment,"keywords":keywords}, status_code=200)
    except:
        a = traceback.format_exc()
        print(a)
        formatted_lines = "Internal server Error: " + str(a.splitlines()[-1])
        return JSONResponse(
            {"message": formatted_lines}, status_code=500
        )
