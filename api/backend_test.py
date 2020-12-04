# importing the requests library 
import requests 
  
# defining the api-endpoint  
PHRASE_COMPLETE_API_ENDPOINT = "http://0.0.0.0:5000/c/phrase_complete_api"
WORD_COMPLETE_API_ENDPOINT = "http://0.0.0.0:5000/c/word_complete_api"
  
word_complete = ["I believe this is the be", "some pe", "th", "is there a right w", "I have ne"]
phrase_complete = ["I believe this is the best", "some people", "this is", "is there a right way", "I have never", "this movie speaks to the", "This is a terrible movie", "this movie needs to be banned", "this is a great movie", "the acting is not that great", "it speaks to me"]
# data to be sent to api
# r = requests.post(url = API_ENDPOINT, json = {'text': sentences[0]}) 

print('*** Testing Phrase Complete API ***')
# test Phrase Complete
for sentence in phrase_complete:
	data = {'text': sentence}
	print(data) 
		
	# sending post request and saving response as response object 
	r = requests.post(url = PHRASE_COMPLETE_API_ENDPOINT, json = data) 
		
	# extracting response text  
	output = r.text 
	print("word_complete is:%s"%output) 

print('-'*40)

print('*** Testing Word Complete API ***')

# test Word Complete
for sentence in word_complete:
	data = {'text': sentence}
	print(data) 
		
	# sending post request and saving response as response object 
	r = requests.post(url = WORD_COMPLETE_API_ENDPOINT, json = data) 
		
	# extracting response text  
	output = r.text 
	print("word_complete is:%s"%output) 
