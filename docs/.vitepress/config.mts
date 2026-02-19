import { defineConfig } from "vitepress";

export default defineConfig({
    title: "livn",
    description: "A testbed for learning to interact with in vitro neural networks",
    base: "/livn/",
    lastUpdated: true,
    cleanUrls: true,
    head: [
        [
            "link",
            {
                rel: "icon",
                href: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>",
            },
        ],
    ],

    themeConfig: {
        nav: [
            { text: "Guide", link: "/guide/getting-started" },
            { text: "Examples", link: "/examples/" },
            { text: "Installation", link: "/installation/" },
        ],

        sidebar: {
            "/guide/": [
                {
                    text: "Introduction",
                    items: [
                        { text: "Getting Started", link: "/guide/getting-started" },
                        { text: "Backends", link: "/guide/backends" },
                        { text: "Datasets", link: "/guide/datasets" },
                    ],
                },
                {
                    text: "Concepts",
                    items: [
                        { text: "Environment", link: "/guide/concepts/env" },
                        { text: "Systems", link: "/guide/concepts/system" },
                        { text: "Models", link: "/guide/concepts/model" },
                        { text: "IO", link: "/guide/concepts/io" },
                        { text: "Decoding", link: "/guide/concepts/decoding" },
                        { text: "Stimulus", link: "/guide/concepts/stimulus" },
                        { text: "Encoding", link: "/guide/concepts/encoding" },
                        { text: "Plasticity", link: "/guide/concepts/plasticity" },
                    ],
                },
                {
                    text: "Systems",
                    items: [
                        { text: "Standard systems", link: "/guide/systems/" },
                        { text: "Generating systems", link: "/guide/systems/generate" },
                        { text: "Tuning systems", link: "/guide/systems/tuning" },
                        { text: "Generating datasets", link: "/guide/systems/sampling" },
                    ],
                },
            ],
            "/examples/": [
                {
                    text: "Examples",
                    items: [
                        { text: "Overview", link: "/examples/" },
                        { text: "Using the Dataset", link: "/examples/dataset" },
                        { text: "Running a Simulation", link: "/examples/simulation" },
                        {
                            text: "Differentiable Simulation",
                            link: "/examples/differentiable",
                        },
                        {
                            text: "Reinforcement Learning",
                            link: "/examples/reinforcement-learning",
                        },
                    ],
                },
            ],
            "/installation/": [
                {
                    text: "Installation",
                    items: [
                        { text: "via uv (recommended)", link: "/installation/" },
                        { text: "Building parallel HDF5", link: "/installation/phdf5" },
                    ],
                },
                {
                    text: "Supercomputers",
                    items: [
                        { text: "TACC Frontera/Vista", link: "/installation/tacc" },
                        { text: "UIUC Campus Cluster", link: "/installation/uiuc-cc" },
                        { text: "NCSA DeltaAI", link: "/installation/ncsa-deltaai" },
                    ],
                },
            ],
        },

        socialLinks: [
            { icon: "github", link: "https://github.com/livn-org/livn" },
        ],

        editLink: {
            pattern: "https://github.com/livn-org/livn/edit/main/docs/:path",
        },

        search: {
            provider: "local",
        },

        footer: {
            message: "Released under the MIT License.",
        },
    },
});
