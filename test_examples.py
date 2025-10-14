from pydantic import BaseModel
import dotenv
from openai import OpenAI

class TestName(BaseModel):
    value: str
    class Config:
        json_schema_extra = {
            "examples": [
                {"value": "Ivan"},
                {"value": "Roman"},
                {"value": "Ignat"},
                {"value": "Petr"},
                {"value": "Oleg"}
            ]
        }


def test_name_schema():
    client = OpenAI(api_key=dotenv.get_key(".env", "OPENAI_API_KEY"))
    print(client.chat.completions.parse(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that creates names.",
            },
            {
                "role": "user",
                "content": "Create a random name.",
            },
        ],
        response_format=TestName,
    ).choices[0].message.parsed)
    
    print(client.chat.completions.parse(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that creates names.",
            },
            {
                "role": "user",
                "content": "Create a random name. Here are some examples: Ivan, Roman, Ignat, Petr, Oleg.",
            },
        ],
        response_format=TestName,
    ).choices[0].message.parsed)


if __name__ == "__main__":
    test_name_schema()