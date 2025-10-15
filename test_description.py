from pydantic import BaseModel, Field
import dotenv
from openai import OpenAI

class TestDescription(BaseModel):
    value_one: str = Field(..., description="The number mentioned in the prompt.")
    value_two: str = Field(..., description="The country mentioned in the prompt.")


class TestNoDescription(BaseModel):
    value_one: str
    value_two: str

def test_description():
    client = OpenAI(api_key=dotenv.get_key(".env", "OPENAI_API_KEY"))
    texts = ("Paris 39 Rome Brazil Ruslan", "Berlin 42 Germany Alice", "Tokyo 27 Japan Bob", "Moscow 23 USA Xie", "Taipei 30 China Josephine")
    system_message =  {
                    "role": "system",
                    "content": "You are a helpful assistant extracts information according to a template.",
                      }
    model = "gpt-5-nano"
    for text in texts:
        print(client.chat.completions.parse(
            model=model,
            messages=[
                system_message,
                {
                    "role": "user",
                    "content": text,
                },
            ],
            response_format=TestDescription,
            ).choices[0].message.parsed)
        print(client.chat.completions.parse(
            model=model,
            messages=[
                system_message,
                {
                    "role": "user",
                    "content": text,
                },
            ],
            response_format=TestNoDescription,
            ).choices[0].message.parsed)

if __name__ == "__main__":
    test_description()