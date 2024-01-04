import express from 'express';
import open from 'open';
import fs from 'fs';

const app = express();
const PORT = 3000;

app.post('/run',(req,res)=>{
    const command = 'cd ../Console_App && conda activate ml && python main.py';
    const command2 = 'cd ../Console_App && conda activate ml && python flowchart_maker.py'
    const scriptContent = `start cmd /k "${command}"`;
    const scriptContent2 = `start cmd /k "${command2}`
    fs.writeFileSync('runScripts.bat', scriptContent);
    fs.writeFileSync(`runScripts2.bat`,scriptContent2)
  open('runScripts.bat', { wait: false });
  open('runScripts2.bat',{wait: false})
})

    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
    });