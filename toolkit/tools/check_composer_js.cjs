// Static assembly only; capture exports from the actual browser composer code.
const fs=require('fs'),vm=require('vm'),path=require('path');
const root=path.resolve(__dirname,'..');
const listeners={};
const context={console,window:null,TK:{},Pages:{},Filter:{task:'',noPre:false},
  R:{esc:String,codebox:s=>s},localStorage:{getItem:()=>null,setItem:()=>{}},
  document:{addEventListener:(k,fn)=>{listeners[k]=fn;}},setTimeout};
context.window=context;vm.createContext(context);
for(const file of ['assets/js/generated/composer.js','assets/js/composer.js'])
  vm.runInContext(fs.readFileSync(path.join(root,file),'utf8'),context,{filename:file});
const books={};
for(const [task,recipe] of Object.entries(context.TK.composer.recipes)) {
  const book=context.Composer.assemble(task);books[task]=book;
  if(!context.Composer.render(task).includes('Unduh .ipynb'))throw Error('render: '+task);
  const st=context.Composer.state(task),original=st.selected.slice();
  for(const selected of recipe.choices.filter(k=>k.startsWith('model.'))) {
    st.selected=[selected];
    const custom=context.Composer.assemble(task);
    if(!custom.metadata.toolkit.resolved.includes(selected))throw Error('missing: '+selected);
  }
  st.selected=original;
  st.config='{bad json';
  let rejected=false;try{context.Composer.assemble(task);}catch(_){rejected=true;}
  if(!rejected)throw Error('Invalid JSON accepted');
  st.config=JSON.stringify(recipe.config);
  st.selected=['model.invalid'];rejected=false;
  try{context.Composer.assemble(task);}catch(_){rejected=true;}
  if(!rejected)throw Error('Incompatible component accepted');
  st.selected=original;
}
const st=context.Composer.state('retrieval');st.selected=['model.rrf'];
const rrf=context.Composer.assemble('retrieval');
if(!['model.bm25','model.e5'].every(k=>rrf.metadata.toolkit.resolved.includes(k)))throw Error('RRF prerequisites');
if(!rrf.cells[1])throw Error('No cells');
context.Composer.noPreChanged(true);
for(const task of Object.keys(books)) {
  const selected=context.Composer.state(task).selected;
  if(selected.some(k=>context.Composer.resolve([k]).some(d=>context.TK.composer.components[d].gpu)))throw Error('GPU filter');
}
process.stdout.write(JSON.stringify(books));
