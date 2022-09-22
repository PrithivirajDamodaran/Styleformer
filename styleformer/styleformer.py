class Styleformer():

  def __init__(
      self,
      style=0,
      ctf_model_tag="prithivida/informal_to_formal_styletransfer",
      ftc_model_tag="prithivida/formal_to_informal_styletransfer",
      atp_model_tag="prithivida/active_to_passive_styletransfer",
      pta_model_tag="prithivida/passive_to_active_styletransfer",
      adequacy_model_tag="prithivida/parrot_adequacy_model", 
  ):
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    from styleformer import Adequacy

    self.style = style
    self.adequacy = adequacy_model_tag and Adequacy(model_tag=adequacy_model_tag)
    self.model_loaded = False

    if self.style == 0:
      self.ctf_tokenizer = AutoTokenizer.from_pretrained(ctf_model_tag)
      self.ctf_model = AutoModelForSeq2SeqLM.from_pretrained(ctf_model_tag)
      print("Casual to Formal model loaded...")
      self.model_loaded = True
    elif self.style == 1:
      self.ftc_tokenizer = AutoTokenizer.from_pretrained(ftc_model_tag)
      self.ftc_model = AutoModelForSeq2SeqLM.from_pretrained(ftc_model_tag)
      print("Formal to Casual model loaded...")
      self.model_loaded = True  
    elif self.style == 2:
      self.atp_tokenizer = AutoTokenizer.from_pretrained(atp_model_tag)
      self.atp_model = AutoModelForSeq2SeqLM.from_pretrained(atp_model_tag)
      print("Active to Passive model loaded...")  
      self.model_loaded = True
    elif self.style == 3:
      self.pta_tokenizer = AutoTokenizer.from_pretrained(pta_model_tag)
      self.pta_model = AutoModelForSeq2SeqLM.from_pretrained(pta_model_tag)
      print("Passive to Active model loaded...")        
      self.model_loaded = True
    else:
      print("Only CTF, FTC, ATP and PTA are supported in the pre-release...stay tuned")

  def transfer(self, input_sentence, inference_on=-1, quality_filter=0.95, max_candidates=5):
      if self.model_loaded:
        if inference_on == -1:
          device = "cpu"
        elif inference_on >= 0 and inference_on < 999:
          device = "cuda:" + str(inference_on)
        else:  
          device = "cpu"
          print("Onnx + Quantisation is not supported in the pre-release...stay tuned.")

        if self.style == 0:
          output_sentence = self._casual_to_formal(input_sentence, device, quality_filter, max_candidates)
          return output_sentence
        elif self.style == 1:
          output_sentence = self._formal_to_casual(input_sentence, device, quality_filter, max_candidates)
          return output_sentence
        elif self.style == 2:
          output_sentence = self._active_to_passive(input_sentence, device)
          return output_sentence        
        elif self.style == 3:
          output_sentence = self._passive_to_active(input_sentence, device)
          return output_sentence           
      else:
        print("Models aren't loaded for this style, please use the right style during init")  


  def _formal_to_casual(self, input_sentence, device, quality_filter, max_candidates):
      ftc_prefix = "transfer Formal to Casual: "
      src_sentence = input_sentence
      input_sentence = ftc_prefix + input_sentence
      input_ids = self.ftc_tokenizer.encode(input_sentence, return_tensors='pt')
      self.ftc_model = self.ftc_model.to(device)
      input_ids = input_ids.to(device)
      
      preds = self.ftc_model.generate(
          input_ids,
          do_sample=True, 
          max_length=32, 
          top_k=50, 
          top_p=0.95, 
          early_stopping=True,
          num_return_sequences=max_candidates)
     
      gen_sentences = set()
      for pred in preds:
        gen_sentences.add(self.ftc_tokenizer.decode(pred, skip_special_tokens=True).strip())

      adequacy_scored_phrases = self.adequacy.score(src_sentence, list(gen_sentences), quality_filter, device)
      ranked_sentences = sorted(adequacy_scored_phrases.items(), key = lambda x:x[1], reverse=True)
      if len(ranked_sentences) > 0:
        return ranked_sentences[0][0]
      else:
        return None

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

  def _passive_to_active(self, input_sentence, device):
      pta_prefix = "transfer Passive to Active: "
      src_sentence = input_sentence
      input_sentence = pta_prefix + input_sentence
      input_ids = self.pta_tokenizer.encode(input_sentence, return_tensors='pt')
      self.pta_model = self.pta_model.to(device)
      input_ids = input_ids.to(device)
      
      preds = self.pta_model.generate(
          input_ids,
          do_sample=True, 
          max_length=32, 
          top_k=50, 
          top_p=0.95, 
          early_stopping=True,
          num_return_sequences=1)
     
      return self.pta_tokenizer.decode(preds[0], skip_special_tokens=True).strip()      

    
