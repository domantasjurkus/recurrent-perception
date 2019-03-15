# set terminal pngcairo  transparent enhanced font "arial,8" fontscale 1.0 size 512, 280 
# set output 'transparent.1.png'
set clip two
set style fill   solid 1.00 noborder
set key title "Gaussian Distribution" center
set key fixed left top vertical Left reverse enhanced autotitle nobox
set key noinvert samplen 1 spacing 1 width 0 height 0 
set style increment default
set style data lines
set style function filledcurves y1=0
set title "Solid filled curves" 
set xrange [ -5.00000 : 5.00000 ] noreverse nowriteback
set x2range [ * : * ] noreverse writeback
set yrange [ 0.00000 : 1.00000 ] noreverse nowriteback
set y2range [ * : * ] noreverse writeback
set zrange [ * : * ] noreverse writeback
set cbrange [ * : * ] noreverse writeback
set rrange [ * : * ] noreverse writeback
unset colorbox
Gauss(x,mu,sigma) = 1./(sigma*sqrt(2*pi)) * exp( -(x-mu)**2 / (2*sigma**2) )
d1(x) = Gauss(x, 0.5, 0.5)
d2(x) = Gauss(x,  2.,  1.)
d3(x) = Gauss(x, -1.,  2.)
save_encoding = "utf8"
plot d1(x) fs solid 1.0 lc rgb "forest-green" title "Î¼ =  0.5 Ïƒ = 0.5",      d2(x) lc rgb "gold" title "Î¼ =  2.0 Ïƒ = 1.0",      d3(x) lc rgb "dark-violet" title "Î¼ = -1.0 Ïƒ = 2.0"