from openai import OpenAI
import logging


class Doc2Query:
    def __init__(self, api_key, system_prompt_path='src/data_pipe/agq/prompt_template.txt'):
        self.client = OpenAI(api_key=api_key)
        with open(system_prompt_path) as fp:
            self.system_prompt = fp.read()

    def generate_queries(self, passage, claim):
        try:
            user_prompt = f"[CLAIM]\n{claim}\n\n[PASSAGE]\n{passage}\n\n[QUESTIONS]"
            completion = self.client.chat.completions.creavte(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"{self.system_prompt}"},
                    {"role": "user", "content": f"{user_prompt}"}
                ],
                temperature=0.1,
                timeout=60
            )
            response = completion.choices[0].message.content
            return [self.__clean_query(query) for query in response.split("\n") if query and "fact-checking" not in query]

        except Exception as e:
            logging.info("Claim could not be processed due to", e.message)
            raise

    @staticmethod
    def __clean_query(query):
        query = query.lstrip("- ")
        query = query.rstrip("\n")
        query = query.rstrip(" ")
        query = query.lstrip(" ")
        return query
