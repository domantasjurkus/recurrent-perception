# set terminal pngcairo  transparent enhanced font "arial,8" fontscale 1.0 size 512, 280 
# set output 'transparent.1.png'

set clip two
set style fill transparent solid 0.2 border
set title "Distributions of video lengths per class"
set samples 1000
set style line 1 lw 2 lc rgb "blue"

# set key title "Gaussian Distribution" center
# set key fixed left top vertical Left reverse enhanced autotitle nobox
set key noinvert samplen 1 spacing 1 width 0 height 0

set style increment default
set style data lines
set style function filledcurves y1=0
# set title "Solid filled curves" 
set xrange [ 44.00000 : 63.00000 ] noreverse nowriteback
set xlabel "video length (seconds)"
# set x2range [ * : * ] noreverse writeback
set yrange [ 0.00000 : 0.35000 ] noreverse nowriteback
set ylabel "probability"
# set y2range [ * : * ] noreverse writeback
# set zrange [ * : * ] noreverse writeback
# set cbrange [ * : * ] noreverse writeback
# set rrange [ * : * ] noreverse writeback
unset colorbox

Gauss(x,mu,sigma) = 1./(sigma*sqrt(2*pi)) * exp( -(x-mu)**2 / (2*sigma**2) )

d0(x) = Gauss(x, 0.5, 0.5)
d1(x) = Gauss(x, 55.43333333, 2.77578719)
d2(x) = Gauss(x,  56.36666667,  2.41459917)
d3(x) = Gauss(x, 50.2,  1.66162024)
d4(x) = Gauss(x, 45,  0)
d5(x) = Gauss(x, 55.56666667,  1.58665064)

save_encoding = "utf8"
plot d1(x) lc rgb "red" title "pant", \
    d2(x) lc rgb "green" title "shirt", \
    d3(x) lc rgb "#cccc00" title "sweater", \
    d4(x) lc rgb "black" title "towel", \
    d5(x) lc rgb "blue" title "tshirt"