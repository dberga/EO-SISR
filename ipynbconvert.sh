if [ -z $1 ]
then
FORMAT='html'
else
FORMAT=$1
fi

for file in *.ipynb
do
jupyter nbconvert --to $FORMAT $file;
done
