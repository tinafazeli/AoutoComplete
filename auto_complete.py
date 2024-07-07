from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# مسیر مدل و توکنایزر
model_path = "./model"
tokenizer_path = "./tokenizer"

# بارگذاری مدل و توکنایزر
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# تولید جمله بر اساس کلمه ورودی
def generate_sentence(word):
    # توکنایز کردن کلمه ورودی
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=32)
    
    # تولید جمله با استفاده از مدل
    outputs = model.generate(
        inputs["input_ids"],
        max_length=132,  # طول جملات تولید شده
        num_beams=5,    # استفاده از جستجوی چند پرتوی
        early_stopping=False,  # ادامه تولید تا رسیدن به طول کامل
        no_repeat_ngram_size=2  # جلوگیری از تکرار n-gramها
    )
    
    # دی‌کد کردن خروجی مدل به جمله
    sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sentence

# مثال استفاده
word = "پد"
sentence = generate_sentence(word)
print(f"کلمه: {word}\nجمله: {sentence}")


