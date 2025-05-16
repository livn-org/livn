import React from 'react';
import { createRoot } from 'react-dom/client';
import NeuralVisualization from './NeuralVisualization';

export function render(el, props) {
    const root = createRoot(el);
    root.render(React.createElement(NeuralVisualization, props));
    return () => root.unmount();
} 