# from SuppSmartFunctions import * 
from upgrade_querySearch import *
import streamlit as st



st.image("suppsmart_banner.jpg", use_column_width=True)
st.subheader('getting smart about dietary supplements')


query = st.text_input('Enter your query:', 'I want something to help me sleep')
topN = st.text_input('Limit your search to top results', 5)
topN = int(topN)

if st.button("'supp me"):
	recList, recTable,searchQuery = wrapperSimilarSupplements(query,topN)
	st.table(recList)
	#st.pyplot( plotCosinSim(recTable))

supp = st.text_input('Enter your query:', 'Melatonin')
nTopics = st.text_input('# of Topics', 3)
nTopics = int(nTopics)

if st.button("test"):
	st.altair_chart(topicModelSupp(sup,nTopics)) #works well, new tab
	#st.pyplot(topicModelSupp("Tryptophan",nTopics))
	#st.markdown(topicModelSupp(supp,nTopics)) #not great
	#plt.show()



st.markdown('')			 
st.markdown('_This supplement explorer is a work of Data Science._')
st.markdown('It is not intended as a substitute for medical expertise. \
			 Consult a physician, registered dietician or other healthcare professional. \
			 The results here should not be taken as an indication or suggestion of any supplement as safe or effective.')

