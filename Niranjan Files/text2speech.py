import requests
import re

def text_to_speech(transcript):
    headers = {
        'AUTHORIZATION': 'e6fc7b189958464f97679eebae063093',
        'X-USER-ID': '5HCUqkHMJWMFhGbD2t9tE67Zri33',
        'accept': 'text/event-stream',
        'content-type': 'application/json',
    }

    json_data = {
        'text': transcript,
        'voice': 's3://voice-cloning-zero-shot/f5e6a3b1-ed91-47eb-95c8-b2ccdc244bf5/dumbledore/manifest.json',
        'output_format': 'mp3',
        'voice_engine': 'PlayHT2.0',
        'speed': 0.8,
    }

    response = requests.post('https://api.play.ht/api/v2/tts', headers=headers, json=json_data)
    
    # Define a regular expression pattern to match the URL
    url_pattern = r'url":\s*"([^"]*)"'

    # Use regex to find the URL in the text
    url_match = re.search(url_pattern, response.text)

    if url_match:
        # Extract the URL from the matched group
        url = url_match.group(1)
        print("Extracted URL:", url)
        
        # Download the audio file
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Specify the file path where you want to save the audio file
                file_path = r'public\audio.mp3'  # You can change the filename if needed
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                #print(f"Audio file downloaded successfully and saved as '{file_path}'.")
            else:
                print(f"Failed to download the audio file. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error downloading the audio file: {e}")
    else:
        print("URL not found in the text.")

    return file_path