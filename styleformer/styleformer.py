class Styleformer():

  def __init__(self,  style=0):
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    from styleformer import Adequacy

    self.style = style
    self.adequacy = Adequacy()

    if self.style == 0:
      self.ctf_tokenizer = AutoTokenizer.from_pretrained("prithivida/informal_to_formal_styletransfer")
      self.ctf_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/informal_to_formal_styletransfer")
      print("Casual to Formal model loaded...")
    elif self.style == 2:
      self.atp_tokenizer = AutoTokenizer.from_pretrained("prithivida/active_to_passive_styletransfer")
      self.atp_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/active_to_passive_styletransfer")
      print("Active to Passive model loaded...")  
    else:
      print("Only Casual to Formal and Active to Passive is supported in the pre-release...stay tuned")

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
      elif self.style == 2:
        output_sentence = self._active_to_passive(input_sentence, device)
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

  def _active_to_passive(self, input_sentence, device):
      atp_prefix = "transfer Active to Passive: "
      src_sentence = input_sentence
      input_sentence = atp_prefix + input_sentence
      input_ids = self.atp_tokenizer.encode(input_sentence, return_tensors='pt')
      self.atp_model = self.atp_model.to(device)
      input_ids = input_ids.to(device)
      
      preds = self.atp_model.generate(
          input_ids,
          do_sample=True, 
          max_length=32, 
          top_k=50, 
          top_p=0.95, 
          early_stopping=True,
          num_return_sequences=1)
     
      return self.atp_tokenizer.decode(preds[0], skip_special_tokens=True).strip()

    