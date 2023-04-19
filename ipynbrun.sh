for file in *.ipynb
do
jupyter nbconvert --to 'python' --execute $file;
done
