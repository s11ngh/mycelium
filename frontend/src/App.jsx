import { useState } from "react";
import "./App.css";
import Hero from "./hero";

function App() {
  const [count, setCount] = useState(0);

  return <Hero></Hero>;
}

export default App;
