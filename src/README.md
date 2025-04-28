### Todo List

* ✅ Write the RAG baseline
* ✅ Write the finetuning script
* ✅ Generate prospective questions using Gemini
* Use HF tokenizer to tokenizer the gutenberg dataset
* configure the GPT2 model according to https://huggingface.co/learn/llm-course/en/chapter7/6
* ✅ Run the model over a long book and compute the perplexity (sanity check)
* ✅ Create a tokenized dataset of english fiction books 
* ✅ List down the token margins of each book
* ✅ write a training script
* ✅ write perplexity script
* ✅ Do a full run
* ✅ Make a plot

### Checklist

* "Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered." -- set "use_sliding_window"  to False
(see https://github.com/huggingface/transformers/pull/36316)
* Check for which modules LORA is applied (https://huggingface.co/docs/peft/en/package_reference/lora)
* ✅ Generated questions
* ✅ RAG 
* ✅ Variable names (alright for now)