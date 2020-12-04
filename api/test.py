text = ''' this is 
text yo   '''

print(text.strip())

print(' '.join([line.strip() for line in text]))
