from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print('loading tokenizer and model')
tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
print('loading model')
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
print('generating')
for i in range(3):
    with open("prompt_"+str(i)+".txt", "r") as f:
        print('===', i)
        prompt = f.read()
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs)
        print(tokenizer.decode(outputs[0]))
print('done')
