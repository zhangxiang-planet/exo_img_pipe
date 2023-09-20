# use DP3

# for i in *.MS
# do
	# Flag bad antennas
	# DP3 DPPP-flagantenna.parset msin=$i
        # calpipe restore_flag.toml $i
	# # Flag RFI
	# DP3 DPPP-aoflagger.parset msin=$i

	# # Calibration using A team model
	# calpipe calibrator.toml $i

	# cp ${i}/instrument_ddecal.h5 ${i}/instrument_ddecal.h5.bak

	# Decompress to inspect with CASA
	# DP3 DPPP-convert.parset msin=SB268.MS msout=SB268_decomp.MS

# done

for i in {08..19}
do
	DP3 DPPP-flagant.parset msin=SB${i}?.MS msout=MSB${i}.MS

	DP3 DPPP-flagchan.parset msin=MSB${i}.MS

	DP3 DPPP-aoflagger.parset msin=MSB${i}.MS

	calpipe calibrator.toml MSB${i}.MS

	# python cal_jones_GTBD_noinv.py
done
