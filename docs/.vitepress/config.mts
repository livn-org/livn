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
            { text: "Installation", link: "/installation/" },
            { text: "Guide", link: "/guide/getting-started" },
            { text: "Systems", link: "/systems/" },
            { text: "Examples", link: "/examples/" },
        ],

        sidebar: {
            "/guide/": [
                {
                    text: "Introduction",
                    items: [
                        { text: "Getting Started", link: "/guide/getting-started" },
                        { text: "Backends", link: "/guide/backends" },
                    ]
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
                    ],
                },
                {
                    text: "Advanced",
                    items: [
                        { text: "Plasticity", link: "/guide/advanced/plasticity" },
                        { text: "Decoding pipelines", link: "/guide/advanced/decoding-pipelines" },
                        { text: "Distributed Environment", link: "/guide/advanced/distributed" },
                        { text: "Gymnasium Integration", link: "/guide/advanced/gymnasium" },
                    ],
                },
            ],
            "/systems/": [
                {
                    text: "Systems",
                    items: [
                        { text: "livn systems", link: "/systems/" },
                        { text: "Datasets", link: "/systems/datasets" },
                        { text: "Generating systems", link: "/systems/generate" },
                        { text: "Tuning systems", link: "/systems/tuning" },
                        { text: "Generating datasets", link: "/systems/sampling" },
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
                            text: "STDP Training",
                            link: "/examples/stdp-training",
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
