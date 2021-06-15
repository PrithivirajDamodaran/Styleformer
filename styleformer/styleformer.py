class Styleformer():

  def __init__(self,  style=0):
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    from adequacy import Adequacy
    
    self.style = style
    self.adequacy = Adequacy()

    if self.style == 0:
      self.ctf_tokenizer = AutoTokenizer.from_pretrained("prithivida/informal_to_formal_styletransfer")
      self.ctf_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/informal_to_formal_styletransfer")
      print("Casual to Formal model loaded...")
    else:
      print("Only Casual to Formal is supported in the pre-release...stay tuned")

  def transfer(self, input_sentence, inference_on=0, quality_filter=0.95, max_candidates=5):
      if inference_on == 0:
        device = "cpu"
      elif inference_on == 1:
        device = "cuda:0"  
      else:  
        device = "cpu"
        print("Onnx + Quantisation is not supported in the pre-release...stay tuned.")

      if self.style == 0:
        output_sentence = self._casual_to_formal(input_sentence, device, quality_filter, max_candidates)
        return output_sentence


  def _casual_to_formal(self, input_sentence, device, quality_filter, max_candidates):
      ctf_prefix = "transfer Casual to Formal: "
      src_sentence = input_sentence
      input_sentence = ctf_prefix + input_sentence
      input_ids = self.ctf_tokenizer.encode(input_sentence, return_tensors='pt')
      self.ctf_model = self.ctf_model.to(device)
      input_ids = input_ids.to(device)
      
      preds = self.ctf_model.generate(
          input_ids,
          do_sample=True, 
          max_length=32, 
          top_k=50, 
          top_p=0.95, 
          early_stopping=True,
          num_return_sequences=max_candidates)
     
      gen_sentences = set()
      for pred in preds:
        gen_sentences.add(self.ctf_tokenizer.decode(pred, skip_special_tokens=True).strip())

      adequacy_scored_phrases = self.adequacy.score(src_sentence, list(gen_sentences), quality_filter, device)
      ranked_sentences = sorted(adequacy_scored_phrases.items(), key = lambda x:x[1], reverse=True)
      if len(ranked_sentences) > 0:
        return ranked_sentences[0][0]
      else:
        return None