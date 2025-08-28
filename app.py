"""App entrypoint for the Derivatives Pricing & Greeks Dashboard

This lightweight wrapper imports and runs the main application defined in
`derivatives_dashboard_app.py`. If that import fails (missing dependencies or
unexpected errors), it shows a minimal Streamlit error page so the user sees
what went wrong instead of a hard crash.
"""

try:
    from derivatives_dashboard_app import main
except Exception as _err:
    import streamlit as st
    import traceback

    # Exception variables are cleared after the except block in Python 3,
    # so capture the traceback text now for later display inside the
    # fallback Streamlit UI.
    _err_traceback = traceback.format_exc()

    def main():
        st.set_page_config(page_title="Derivatives Pricing Dashboard - Error", layout="wide")
        st.title("Derivatives Pricing & Greeks Dashboard")
        st.error("Failed to load the full dashboard. See details below.")
        st.markdown("**Import traceback:**")
        st.code(_err_traceback, language='text')


if __name__ == "__main__":
    main()
