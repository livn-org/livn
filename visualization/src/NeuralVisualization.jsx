import { useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import MorphologyRenderer from './MorphologyRenderer';

function NeuronGroup({ name, neurons, showMorphologies, morphologies }) {
    return (
        <>
            {!showMorphologies && neurons.map((neuron, index) => {
                neuron.refs = neuron.refs || {};
                neuron.refs.soma = useRef();
                return (
                    <mesh
                        key={index}
                        position={[neuron.x, neuron.y, neuron.z]}
                        ref={neuron.refs.soma}
                    >
                        <sphereGeometry args={[0.08, 32, 16]} />
                        <meshStandardMaterial color={name === "PYR" ? "red" : "green"} />
                    </mesh>
                );
            })}
            {showMorphologies && morphologies && morphologies[name] && (
                <MorphologyRenderer
                    morphologies={morphologies}
                    neurons={neurons}
                    population={name}
                />
            )}
        </>
    );
}

function MEA3D({ neurons }) {
    return (
        <>
            {neurons.map((neuron, index) => {
                neuron.refs = neuron.refs || {};
                neuron.refs.mea = useRef();
                return (
                    <group key={index}>
                        <mesh
                            position={[neuron.x, neuron.y + 4, neuron.z]}
                            rotation={[Math.PI, 0, 0]}
                        >
                            <cylinderGeometry args={[0.2, 0.2, 2, 32]} />
                            <meshStandardMaterial color="gold" />
                        </mesh>

                        <mesh
                            position={[neuron.x, neuron.y + 2, neuron.z]}
                            rotation={[Math.PI, 0, 0]}
                        >
                            <coneGeometry args={[0.2, 2, 32]} />
                            <meshStandardMaterial color="gold" />
                        </mesh>

                        <mesh
                            position={[neuron.x, neuron.y + 5, neuron.z]}
                        >
                            <boxGeometry args={[0.8, 0.1, 0.8]} />
                            <meshStandardMaterial color="grey" />
                        </mesh>

                        <mesh
                            position={[neuron.x, neuron.y + 5.1, neuron.z]}
                            ref={neuron.refs.mea}
                        >
                            <cylinderGeometry args={[0.3, 0.3, 0.1, 32]} />
                            <meshStandardMaterial color="grey" />
                        </mesh>
                    </group>
                );
            })}
        </>
    );
}

function Dish() {
    return (
        <>
            <mesh position={[5, 1.5, 5]}>
                <cylinderGeometry args={[8., 8., 5, 32]} />
                <meshBasicMaterial wireframe={false} opacity={0.1} transparent={true} color='grey' />
            </mesh>
            <mesh position={[5, 1., 5]}>
                <cylinderGeometry args={[7.8, 7.8, 4, 32]} />
                <meshBasicMaterial wireframe={false} opacity={0.1} transparent={true} color='blue' />
            </mesh>
        </>
    );
}

function NeuralVisualization({ data, show_mea, show_dish, show_morphologies, morphologies }) {
    if (!data || !data.populations) {
        return <div>No data provided</div>;
    }

    const simulation = {};
    Object.keys(data.populations).forEach((population) => {
        simulation[population] = {
            name: population,
            neurons: data.populations[population].neurons.map(({ i, x, y, z }) => ({
                i,
                x: x / 4000. * 10,
                y: z / 450. * 3, // y-up!
                z: y / 4000. * 10,
                refs: {}
            }))
        };
    });

    return (
        <div style={{ width: '100%', height: '500px' }}>
            <Canvas>
                <PerspectiveCamera makeDefault position={[10, 10, 10]} fov={75} />
                <OrbitControls />
                <ambientLight intensity={0.4} />
                <directionalLight color="white" position={[0, 0, 5]} />
                
                {Object.entries(simulation).map(([name, population]) => (
                    <NeuronGroup 
                        key={name} 
                        {...population} 
                        showMorphologies={show_morphologies}
                        morphologies={morphologies}
                    />
                ))}

                {show_mea && simulation.STIM && (
                    <MEA3D neurons={simulation.STIM.neurons} />
                )}

                {show_dish && <Dish />}
            </Canvas>
        </div>
    );
}

export default NeuralVisualization; 