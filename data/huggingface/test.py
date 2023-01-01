from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
for i in range(3):
    with open("prompt_"+str(i)+".txt", "r") as f:
        print('===', i)
        prompt = f.read()
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs)
        print(tokenizer.decode(outputs[0]))
print('done')
