import os
import re
import json
import base64
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
import xml.etree.ElementTree as ET
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

import time ## added.
import atexit ## added.
elapsed_times = {} ##added.
# word = 'Artificial Intelligence'
word = '(semiconductor !image !photo !poster !ppt !review) '
OUTPUT_PATH = Path(__file__).parent / f'outputs/sample/sai_elapsed_times_{word}.json'

def save_elapsed_times():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)  # Create directory if not exists
    with open(OUTPUT_PATH, "w") as f:
        json.dump(elapsed_times, f, indent=4)
    print(f"✅ Elapsed times saved to {OUTPUT_PATH}") ##Yesim added.


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
BASE_URL = "https://apigateway.kisti.re.kr/openapicall.do"
TOKEN_REQUEST_URL = "https://apigateway.kisti.re.kr/tokenrequest.do"
TOKEN_EXPIRY_BUFFER = timedelta(minutes=1)

class AESCipher: #암호화 (hw)
    """A consolidated class for handling AES-CBC encryption."""
    def __init__(self, auth_key: str):
        if len(auth_key) != 32:
            raise ValueError("API key must be 32 bytes.")
        self.key = auth_key.encode('utf-8')
        self.block_size = AES.block_size
        self.iv = 'jvHJ1EFA0IXBrxxz'.encode('utf-8')

    def encrypt(self, plain_text: str) -> str:
        """Encrypts plaintext using AES-CBC mode."""
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        padded_bytes = pad(plain_text.encode('utf-8'), self.block_size)
        encrypted_bytes = cipher.encrypt(padded_bytes)
        return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')

class CredentialManager: #API, Key, Token 관리 Class (hw)
    """A class for synchronously managing API credentials."""
    def __init__(self, credentials_path: Path):
        self.credentials_path = credentials_path
        self.credentials = {}
        self._load_credentials()
        self.aes_cipher = AESCipher(self.auth_key)
        self.lock = Lock()

    def _load_credentials(self):
        """Loads credentials from a file."""
        try:
            with open(self.credentials_path, "r", encoding='utf-8') as f:
                self.credentials = json.load(f)
        except FileNotFoundError:
            logging.error(f"Credential file not found: {self.credentials_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Invalid format in credential file: {self.credentials_path}")
            raise

    def _save_credentials(self):
        """Saves credentials from memory to a file."""
        with open(self.credentials_path, "w", encoding='utf-8') as f:
            json.dump(self.credentials, f, indent=4)

    @property
    def mac_address(self) -> str: return self.credentials.get("mac_address")
    @property
    def auth_key(self) -> str: return self.credentials.get("auth_key")
    @property
    def client_id(self) -> str: return self.credentials.get("client_id")
    @property
    def access_token(self) -> str: return self.credentials.get("access_token")
    @property
    def refresh_token(self) -> str: return self.credentials.get("refresh_token")

    def _is_token_valid(self, token_expiry_str: str) -> bool:
        """Checks if the token is expired."""
        if not token_expiry_str:
            return False
        try:
            # Handle different datetime formats
            if '.' in token_expiry_str:
                expire_time = datetime.strptime(token_expiry_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
            else:
                expire_time = datetime.strptime(token_expiry_str, "%Y-%m-%d %H:%M:%S")
            return datetime.now() < (expire_time - TOKEN_EXPIRY_BUFFER)
        except (ValueError, TypeError):
            logging.warning(f"Invalid expiration time format: {token_expiry_str}")
            return False

    def _update_tokens(self, token_data: dict):
        """Updates the internal state with new token information and saves it to the file."""
        self.credentials.update(token_data)
        self._save_credentials()
        logging.info("Token information updated successfully.")

    def _request_new_tokens(self, session: requests.Session):
        """Requests new access and refresh tokens using the API key."""
        logging.info("Requesting new access/refresh tokens.")
        current_time_str = ''.join(re.findall(r"\d", datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        plaintext_payload = json.dumps({"datetime": current_time_str, "mac_address": self.mac_address}).replace(" ", "")

        # Use the consolidated AESCipher class
        encrypted_payload = self.aes_cipher.encrypt(plaintext_payload)

        params = {'client_id': self.client_id, 'accounts': encrypted_payload}
        with session.get(TOKEN_REQUEST_URL, params=params) as response:
            response.raise_for_status()
            new_credentials = response.json()
            self._update_tokens(new_credentials)

    def _refresh_access_token(self, session: requests.Session):
        """Requests a new access token using the refresh token."""
        logging.info("Renewing access token using the refresh token.")
        params = {'refresh_token': self.refresh_token, 'client_id': self.client_id}
        with session.get(TOKEN_REQUEST_URL, params=params) as response:
            response.raise_for_status()
            new_credentials = response.json()
            self._update_tokens(new_credentials)
    
    def get_access_token(self, session: requests.Session) -> str:
        """Synchronously returns a valid access token."""
        with self.lock:
            if self._is_token_valid(self.credentials.get("access_token_expire")):
                return self.access_token

            logging.warning("Access token has expired. Attempting to renew.")
            if self._is_token_valid(self.credentials.get("refresh_token_expire")):
                self._request_new_tokens(session)
            else:
                logging.warning("Refresh token has also expired. Requesting a new set of tokens.")
                self._request_new_tokens(session)
            
            return self.access_token

class ScienceONAPIClient: #논문 검색 Class (hw)
    """A synchronous client for the ScienceON API."""
    def __init__(self, credentials_path: Path):
        self.credential_manager = CredentialManager(credentials_path)
        self.session = requests.Session()

    def close_session(self):
        """Closes the requests session."""
        self.session.close()

    @staticmethod
    def _parse_search_response(xml_text: str, fields: list[str]) -> list[dict]:
        """Parses the API XML response to extract specified fields."""
        root = ET.fromstring(xml_text)
        records = []
        field_map = {
            'CN': {'metaCode': 'CN'},
            'title': {'metaName': '논문명'},
            'abstract': {'metaName': '초록'},
            'author': {'metaName': '저자'},
            'link': {'metaName': 'ScienceON상세링크'},
            'publisher': {'metaName': '출판사(발행기관)'},
            'journal': {'metaName': '저널명'},
            'year': {'metaName': '발행년'}
        }
        
        requested_fields = {k: v for k, v in field_map.items() if k in fields}
        
        record_list = root.find('recordList')
        if record_list is None:
            return records

        for record in record_list.findall('record'):
            record_dict = {}
            for item in record.findall('item'):
                text = item.text.strip() if item.text else ""
                for field_name, criteria in requested_fields.items():
                    if all(item.attrib.get(k) == v for k, v in criteria.items()):
                        record_dict[field_name] = text
                        break
            if record_dict:
                records.append(record_dict)
        return records

    def search_articles(self, query: str, cur_page: int = 1, row_count: int = 10,
                              fields: list[str] = None) -> list[dict]:
        """
        Synchronously searches for articles.
        
        :param query: The search keyword (e.g., "Quantum Mechanics").
        :param cur_page: The current page number.
        :param row_count: The number of results per page.
        :param fields: A list of fields to retrieve. 
                       Default: ['title', 'author', 'abstract', 'CN'].
                       Available: 'title', 'abstract', 'author', 'link', 'publisher', 'journal', 'year', 'CN'.
        :return: A list of dictionaries containing the search results.
        """
        if fields is None:
            fields = ['title', 'author', 'abstract', 'CN']

        access_token = self.credential_manager.get_access_token(self.session)
        
        params = {
            'client_id': self.credential_manager.client_id,
            'token': access_token,
            'version': '1.0',
            'action': 'search',
            'target': 'ARTI',
            'searchQuery': f'{{"BI":"{query}"}}',
            'sortField': '',
            'curPage': str(cur_page),
            'rowCount': str(row_count),
            'session_id': '',
            'include': '',
            'grouping': ''
        }

        try:
            with self.session.get(BASE_URL, params=params) as response:
                response.raise_for_status()
                xml_text = response.text
                return self._parse_search_response(xml_text, fields)
        except requests.RequestException as e:
            logging.error(f"An error occurred during the API request: {e}")
            return []


def main():
    """Main execution function."""
    start_time = time.perf_counter() ## added.
    
    # Dependencies: pip install requests pycryptodome
    # Assuming the 'configs' folder is in the same directory as the script.
    credentials_path = Path(__file__).parent / './configs/scienceon_api_credentials.json'
    client = ScienceONAPIClient(credentials_path=credentials_path)

    try:
        # print("\n--- Search for 'Big Data' (default fields, 10 results) ---")
        
        # # Search with default fields
        # #search_results = client.search_articles('Quantum Mechanics', row_count=10)
        # search_results=client.search_articles('Big Data')
        # output_path = Path(__file__).parent / 'keyword_big_data_search_results.json'
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(search_results, f, ensure_ascii=False, indent=2)
            
        # print(f"Search results saved to '{output_path}'")

        
        print("\n--- Search for 'Big Data' (all fields, 50 results) ---")

        output_path_all = Path(__file__).parent / f'outputs/sample/search_results_all_fields_{word}.json'
        all_fields = ['title', 'abstract', 'author', 'link', 'publisher', 'journal', 'year', 'CN']
        #full_search_results = client.search_articles('Artificial Intelligence', row_count=3, fields=all_fields)
        full_search_results = client.search_articles(word, row_count=50, fields=all_fields)
        with open(output_path_all, 'w', encoding='utf-8') as f:
            json.dump(full_search_results, f, ensure_ascii=False, indent=2)
            
        elapsed = time.perf_counter() - start_time #Yesim added.
        elapsed_times['search_keyword_elapsed_time'] = elapsed_times.get('search_keyword_elapsed_time', 0) + elapsed #Yesim added.
        print(f"search keyword elapsed time: {elapsed:.4f} seconds") #Yesim added.
        print(f"Search results saved to '{output_path_all}'")

    finally:
        client.close_session()

if __name__ == "__main__":
    # Assumes the script is in the same directory as the 'configs' folder, 
    # which contains the scienceon_api_credentials.json file.
    main()
    atexit.register(save_elapsed_times) ##Yesim added.