# importing the requests library 
import requests 
  
# defining the api-endpoint  
API_ENDPOINT = "http://127.0.0.1:5000/c/phrase_complete_api"
  
sentences = ["I believe this is the be", "some pe", "th", "is there a right w", "I have ne"]
# data to be sent to api
# r = requests.post(url = API_ENDPOINT, json = {'text': sentences[0]}) 
for i in range(5):
	for sentence in sentences:
		data = {'text': sentence}
		print(data) 
		  
		# sending post request and saving response as response object 
		r = requests.post(url = API_ENDPOINT, json = data) 
		  
		# extracting response text  
		output = r.text 
		print("word_complete is:%s"%output) 
