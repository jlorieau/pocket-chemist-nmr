# PocketChemist - NMR

The pocketchemist NMR plugin is used to process and analyze data from
Nuclear Magnetic Resonance (NMR).

## Contributing

### Procedure for new processing Methods

1. Implement the new procedure in the NMRSpectrum abstract base class

2. Implement metadata updates in the NMRSpectrum concrete subclasses as well
   as implementation details through parameters in the called parent method

3. Implement the CLI for the new process

4. Add tests to verify that the implementation matches an existing processed
   spectrum, if it exists

5. Add documentation for the new CLI functionality and python interface on 
   the new process

6. Increase the minor version


# Credits

- Icons by [Icons8](https://icons8.com/)