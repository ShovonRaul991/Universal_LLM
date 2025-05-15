from streamlit_local_storage import LocalStorage
import streamlit as st
localS = LocalStorage()

itemKey = "camera"
itemValue = "Tarah"
localS.setItem(itemKey, itemValue)

saved_individual = localS.getAll()
st.write(saved_individual)

itemKey = "camera"
local_storage_item = localS.getItem(itemKey)
st.write(local_storage_item)

localS.deleteAll()