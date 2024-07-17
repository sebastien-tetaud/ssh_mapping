mkdir data
cd data/
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2020a_SSH_mapping_NATL60/dc_obs.tar.gz
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2020a_SSH_mapping_NATL60/dc_ref.tar.gz
tar -xvf dc_obs.tar.gz
tar -xvf dc_ref.tar.gz
rm -fr dc_obs.tar.gz
rm -fr dc_obs.tar.gz