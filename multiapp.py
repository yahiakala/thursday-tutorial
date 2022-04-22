"""Framework for running multiple Streamlit applications as a single app."""
# https://icons.getbootstrap.com/
import streamlit as st
from streamlit_option_menu import option_menu

class MultiApp:
    """
    Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """
    def __init__(self):
        self.apps = []

    def add_app(self, title, func, icon_bootstrap):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func,
            "icon": icon_bootstrap
        })

    def run(self):
        # Manually control the icons
        selected = option_menu(
            menu_title=None,  # required
            options=[i['title'] for i in self.apps],  # required
            icons=[i['icon'] for i in self.apps],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        for i in self.apps:
            if i['title'] == selected:
                app = i
        # app = st.selectbox(
        #     'Navigation',
        #     self.apps,
        #     format_func=lambda app: app['title'])

        app['function']()