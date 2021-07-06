from styleformer import Styleformer
import streamlit as st
import numpy as np
import json

class Demo:
    def __init__(self):
        st.set_page_config(
            page_title="Styleformer Demo",
            initial_sidebar_state="expanded"
            )
        self.style_map = {
            #key : (name , style_num)
            'ctf': ('Casual to Formal', 0),
            'ftc': ('Formal to Casual', 1),
            'atp': ('Active to Passive', 2),
            'pta': ('Passive to Active', 3)
            }
        self.inference_map = {
            0: 'Regular model on CPU',
            1: 'Regular model on GPU',
            2: 'Quantized model on CPU'
        }
        with open("streamlit_examples.json") as f:
            self.examples = json.load(f)

    @st.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
    def load_sf(self, style=0):
        sf = Styleformer(style = style)
        return sf
        
    def main(self):
        github_repo = 'https://github.com/PrithivirajDamodaran/Styleformer'
        st.title("Styleformer")
        st.write(f'GitHub Link - [{github_repo}]({github_repo})')
        st.write('A Neural Language Style Transfer framework to transfer natural language text smoothly between fine-grained language styles like formal/casual, active/passive, and many more')

        style_key = st.sidebar.selectbox(
            label='Choose Style',
            options=list(self.style_map.keys()),
            format_func=lambda x:self.style_map[x][0]
            )
        exp = st.sidebar.beta_expander('Knobs', expanded=True)
        with exp:
            inference_on = exp.selectbox(
                label='Inference on',
                options=list(self.inference_map.keys()),
                format_func=lambda x:self.inference_map[x]
                )
            quality_filter = exp.slider(
                label='Quality filter',
                min_value=0.5,
                max_value=0.99,
                value=0.95
                )
            max_candidates = exp.number_input(
                label='Max candidates',
                min_value=1,
                max_value=20,
                value=5
                )
        with st.spinner('Loading model..'):
            sf = self.load_sf(self.style_map[style_key][1])
        input_text = st.selectbox(
            label="Choose an example",
            options=self.examples[style_key]
            )
        input_text = st.text_input(
            label="Input text",
            value=input_text
        )

        if input_text.strip():
            result = sf.transfer(input_text, inference_on=inference_on, quality_filter=quality_filter, max_candidates=max_candidates)
            st.markdown(f'#### Output:')
            st.write('')
            if result:
                st.success(result)
            else:
                st.info('No good quality transfers available !')
        else:
            st.warning("Please select/enter text to proceed")
        


if __name__ == "__main__":
    obj = Demo()
    obj.main()

    
