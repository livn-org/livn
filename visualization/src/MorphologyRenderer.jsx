import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import SharkViewer from '@janelia/sharkviewer';
import * as THREE from 'three';

function MorphologyRenderer({ morphologies, neurons, population }) {
    const { camera, gl } = useThree();
    const sharkviewerRef = useRef(null);
    const neuronsRef = useRef([]);

    useEffect(() => {
        if (!morphologies || !morphologies[population]) return;

        if (!sharkviewerRef.current) {
            sharkviewerRef.current = new SharkViewer({
                renderer: gl,
                camera: camera,
            });

            sharkviewerRef.current.three_colors = [];
            Object.keys(sharkviewerRef.current.colors).forEach(color => {
                sharkviewerRef.current.three_colors.push(new THREE.Color(sharkviewerRef.current.colors[color]));
            });
            sharkviewerRef.current.three_materials = [];
            Object.keys(sharkviewerRef.current.colors).forEach(color => {
                sharkviewerRef.current.three_materials.push(
                    new THREE.MeshBasicMaterial({
                        color: sharkviewerRef.current.colors[color],
                        wireframe: false
                    })
                );
            });
        }

        const swc = morphologies[population];
        neuronsRef.current.forEach(neuron => {
            if (neuron.object) {
                neuron.object.parent.remove(neuron.object);
            }
        });
        neuronsRef.current = neurons.map(neuron => {
            const object = sharkviewerRef.current.createNeuron(swc);
            object.position.set(neuron.x, neuron.y, neuron.z);
            object.scale.set(0.1, 0.1, 0.1);
            return { ...neuron, object };
        });

        return () => {
            neuronsRef.current.forEach(neuron => {
                if (neuron.object) {
                    neuron.object.parent.remove(neuron.object);
                }
            });
        };
    }, [morphologies, neurons, population, camera, gl]);

    return null;
}

export default MorphologyRenderer; 