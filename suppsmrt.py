from SuppSmartFunctions import * 
import streamlit as st





st.markdown("# _supp_'smart")
st.subheader('getting smart about dietary supplements')

st.markdown('')			 
st.markdown('_This supplement explorer is a work of Data Science._')
st.markdown('It is not intended as a substitute for medical expertise. \
			 Consult a physician, registered dietician or other healthcare professional. \
			 The results here should not be taken as an indication or suggestion of any supplement as safe or effective.')



query = st.text_input('Enter your query:', 'I want something to help me sleep')
topN = st.text_input('Limit your search to top results', 5)
topN = int(topN)

if st.button("'supp me"):
	recList, recTable,searchQuery = wrapperSimilarSupplements(query,topN)
	st.table(recList)
	#st.pyplot( plotCosinSim(recTable))
if st.button("Show similarity between query and supplements"):
	_, recTable,_ = wrapperSimilarSupplements(query,topN)
	st.bar_chart(recTable['aggCosSim'])
	

# if st.button("similar words"):
# 	recList, recTable,searchQuery = wrapperSimilarSupplements(query,topN)
# 	st.write(searchQuery)
# 	st.pyplot(queryUMAP(searchQuery,doc2vec_model,umap_wordvecs))



