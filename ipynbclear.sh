for file in *.ipynb
do
jupyter nbconvert --clear-output --inplace $file;
done
