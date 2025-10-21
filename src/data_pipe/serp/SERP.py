# google api key
from datetime import datetime
import json
from typing import List, Dict

import requests

from googleapiclient.discovery import build


class SERP:
    def __init__(self, api_key) -> None:
        self.api_key = api_key       


    def fetch_results(self, query: str, num: int, file_type: str = "html") -> List[Dict]:
        return None


class GoogleCustomSearch(SERP):
    def __init__(self, api_key, engine_id):
        super().__init__(api_key)
        self.engine_id = engine_id
        self.service = self.build_service(api_key)

    def build_service(self, api_key):
        return build("customsearch", "v1", developerKey=api_key)

    def fetch_results(
        self, query: str, filter_date: str = None, num: int = 10, file_type: str = "html"
    ) -> List[Dict]:
        now = datetime.now()
        filter_date = filter_date if filter_date else now.strftime("%Y-%m-%d")
        result = (
            self.service.cse()
            .list(
                q=f"{query} before:{filter_date}",
                cx=self.engine_id,
                num=10,
                fileType=file_type,
                filter=True,
            )
            .execute()
        )

        search_records = []

        if "items" in result.keys() and result["items"]:
            for item in result["items"]:
                search_records.append(item)
        return search_records
    
class SERPER(SERP):
    def __init__(self, api_key, serper_url='https://google.serper.dev/search'):
        super().__init__(api_key)
        self.serper_url = serper_url

    def fetch_results(
        self, query: str, filter_date: str = None, num: int = 10,
    ) -> List[Dict]:

        payload = json.dumps({
        "q": f"{query} before:{filter_date}",
        "num": num
        })
        headers = {
        'X-API-KEY': self.api_key,
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", self.serper_url, headers=headers, data=payload)
        response = json.loads(response.text)

        return response['organic']
