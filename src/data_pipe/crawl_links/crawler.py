import base64
import grequests
from requests.models import Response
from urllib.parse import unquote

# Import Beautiful Soup
from boilerpy3.extractors import Extractor,ArticleExtractor
import logging

 
# Initialize the object with the document

 
# Get the whole body tag
 
# Print each string recursively


class Crawler:

    def __init__(self,extractor:Extractor=ArticleExtractor()) -> None:
        self.headers = {'User-Agent':"Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.4; en-US; rv:1.9.2.2) Gecko/20100316 Firefox/3.6.2"
                        ,'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
                        'Accept-Language': 'en-IE,en-GB;q=0.9,en-US;q=0.8,en;q=0.7'}
        self.extractor = extractor
        self.error_log = []
        
        
    def _exception_handler(self,request, exception):
        response = Response()
        response.status_code = 400
        response._content = exception.args[0]
        response.url = request.url
        return response

    
    def _save_text(self,save_dir,file_name,full_text):
        full_path = save_dir+'/'+file_name+'.txt'
        with open(full_path, 'w+') as file:
            # Writing the string to the file
            file.write(full_text)
    
    def make_request(self,url,headers,timeout,verify):
        request = grequests.get(url,headers=headers,timeout=timeout,verify=verify)
        return request
    
    def crawl_links(self,links,save=False,save_dir=None):
        unquoted_links = {unquote(k).rstrip('#'):v for k,v in links.items()}

        reqs = [self.make_request(url,headers=self.headers,timeout=2,verify = False) for url in list(links.keys())]
        results = {}
        logs = {}
        for resp in grequests.imap(reqs, size=5,exception_handler=self._exception_handler):
            original_url = unquote(resp.history[0].url if resp.history else resp.url)
            if(resp.status_code in [200,203]):
                try:
                    content = self.extractor.get_content(resp.text)
                    index = unquoted_links[original_url]
                    if(content):
                        if(save):                            
                            self._save_text(file_name=str(index),save_dir=save_dir,full_text = content)
                        else:
                            results[index] = (original_url,content)
                except Exception as e:
                    logs[original_url] = {"error_type":"PARSE","error":str(e)}
                    print("[Error][PARSE]",resp.url)
                    print(e)
                    continue
            else:
                logs[original_url] = {"error_type":"NETWORK","error":resp.status_code}
                print("[Error][NETWORK]",resp.url)
                print(resp.status_code)
                print(resp.url)
        self.error_log.append(logs)
        return results
                    
                    

