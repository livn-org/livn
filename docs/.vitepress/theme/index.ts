import DefaultTheme from "vitepress/theme";
import PyodideWidget from "./PyodideWidget.vue";

export default {
    extends: DefaultTheme,
    enhanceApp({ app }) {
        app.component("PyodideWidget", PyodideWidget);
    },
};
