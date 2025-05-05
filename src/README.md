### Todo List

* Train the LoRA adapters on hp1_2 chapters
* Get results for RAG and no FT/RAG on hp1_2 
* ✅ save a list of chapters hp1, hp2, hp1 and hp2 combined
* ✅ Write the RAG baseline
* ✅ Write the finetuning script
* ✅ Generate prospective questions using Gemini
* ✅ Run the model over a long book and compute the perplexity (sanity check)
* ✅ Create a tokenized dataset of english fiction books 
* ✅ List down the token margins of each book
* ✅ write a training script
* ✅ write perplexity script
* ✅ Do a full run
* ✅ Make a plot
* ⏸️ Use HF tokenizer to tokenizer the gutenberg dataset
* ⏸️ configure the GPT2 model according to https://huggingface.co/learn/llm-course/en/chapter7/6

### Checklist

* Are we finetuning correctly? Is LoRA the way to go?
* ✅ "Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered." -- set "use_sliding_window"  to False
(see https://github.com/huggingface/transformers/pull/36316)
* ✅ Check for which modules LORA is applied (https://huggingface.co/docs/peft/en/package_reference/lora) [V_proj, Q_proj]
* ✅ Generated questions
* ✅ RAG 
* ✅ Variable names (alright for now)