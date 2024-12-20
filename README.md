# LLMInit: A Free Lunch from Large Language Models for Selective Initialization of Recommendation

This is the Pytorch Implementation for LLMInit.

Examples: 
(1) run the **LLMInit-Rand** with the **LightGCN** on Amazon-Beauty
'''
python run_recbole.py --opt rand -d amazon-beauty -m ContGCN
'''

(2) run the **LLMInit-Uni** with the **SGL** on Amazon-Beauty
'''
python run_recbole.py --opt uni -d amazon-beauty -m ContSGL
'''

(3) run the **LLMInit-Var** with the **SGCL** on Amazon-Beauty
'''
python run_recbole.py --opt var -d amazon-beauty -m ContSGCL
'''

## Acknowledgement
The structure of this repo is built based on [RecBole](https://github.com/RUCAIBox/RecBole). Thanks for their great work.