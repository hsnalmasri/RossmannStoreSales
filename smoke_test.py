# Sanity check outside of Streamlit run
import importlib.util, pathlib
p = pathlib.Path("streamlit_app.py")
spec = importlib.util.spec_from_file_location("app_main", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("streamlit_app.py imported successfully.")
