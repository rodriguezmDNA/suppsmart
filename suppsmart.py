# from SuppSmartFunctions import * 
from upgrade_querySearch import *
import streamlit as st



st.image("suppsmart_banner.jpg", use_column_width=True)
st.subheader('getting smart about dietary supplements')


query = st.text_input('enter your query:', 'I want something to help me sleep')
topN = st.text_input('limit your search to top results', 5)
topN = int(topN)

supReq = False

if st.button("'supp me"):
	recList, recTable,searchQuery = wrapperSimilarSupplements(query,topN)
	st.table(recList)
	supReq = True

	#st.pyplot( plotCosinSim(recTable))


st.subheader('explore what people have said')
sugSupp = st.text_input('which supplement?', "Caffeine")
nTopics = st.text_input('# of topics', 8)
nTopics = int(nTopics)

if st.button("explore topics"):
	resLDA, resI2W, resCorp = buildTopicModel(sugSupp,"tri",nTopics,"lda")
	tmpTopics = [list(dict(resLDA.show_topic(_)).keys()) for _ in range(nTopics)]
	tmpTopics = ["".join(str(term)) for term in tmpTopics]
	st.table(tmpTopics)

# 	#topicModelSupp(sup,nTopics)) #works well, new tab
# 	#st.pyplot(topicModelSupp("Tryptophan",nTopics))
# 	#st.markdown(topicModelSupp(supp,nTopics)) #not great
# 	#plt.show()


st.markdown('')			 
st.markdown('_This supplement explorer is a work of Data Science._')
st.markdown('It is not intended as a substitute for medical expertise. \
			 Consult a physician, registered dietician or other healthcare professional. \
			 The results here should not be taken as an indication or suggestion of any supplement as safe or effective.')

