from textblob import TextBlob
text = input("type some text ")


blob = TextBlob(text)
y = blob.sentiment
z = str(y)

print(y)
print(z)

split = z.split(",")
print(split)