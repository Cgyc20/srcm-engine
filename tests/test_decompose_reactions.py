from srcm_engine.reactions import HybridReactionSystem
import pytest

class TestReactionDecomposition:
    """Test automatic decomposition of reactions into hybrid SRCM reactions."""
    
    def test_zero_order_reaction(self):
        """Test zero-order reaction: ∅ → A"""
        reactions = HybridReactionSystem(species=["A"])
        
        # Zero-order: creation of A
        reactions.add_reaction_original({}, {"A": 1}, rate=0.5, rate_name="r_create")
        
        # Should generate 1 hybrid reaction
        assert len(reactions.hybrid_reactions) == 1
        
        hr = reactions.hybrid_reactions[0]
        assert hr.label == "r_create_zero"
        assert hr.state_change == {"D_A": 1}
        assert hr.reactants == {}
        assert hr.products == {"D_A": 1}
        
        # Test propensity: should be r * h
        D = {"A": 0}
        C = {"A": 0.0}
        r = {"r_create": 0.5}
        h = 0.1
        
        prop = hr.propensity(D, C, r, h)
        assert prop == 0.5 * 0.1  # r * h
    
    
    def test_first_order_reaction(self):
        """Test first-order reaction: A → B"""
        reactions = HybridReactionSystem(species=["A", "B"])
        
        # First-order: A → B
        reactions.add_reaction_original({"A": 1}, {"B": 1}, rate=2.0, rate_name="r_convert")
        
        # Should generate 1 hybrid reaction
        assert len(reactions.hybrid_reactions) == 1
        
        hr = reactions.hybrid_reactions[0]
        assert hr.label == "r_convert_D"
        assert hr.state_change == {"D_A": -1, "D_B": 1}
        assert hr.reactants == {"D_A": 1}
        assert hr.products == {"D_B": 1}
        
        # Test propensity: should be r * D_A
        D = {"A": 10, "B": 5}
        C = {"A": 0.0, "B": 0.0}
        r = {"r_convert": 2.0}
        h = 0.1
        
        prop = hr.propensity(D, C, r, h)
        assert prop == 2.0 * 10  # r * D_A
    
    
    def test_first_order_degradation(self):
        """Test first-order degradation: A → ∅"""
        reactions = HybridReactionSystem(species=["A"])
        
        # Degradation: A → nothing
        reactions.add_reaction_original({"A": 1}, {}, rate=1.5, rate_name="r_degrade")
        
        # Should generate 1 hybrid reaction
        assert len(reactions.hybrid_reactions) == 1
        
        hr = reactions.hybrid_reactions[0]
        assert hr.label == "r_degrade_D"
        assert hr.state_change == {"D_A": -1}
        assert hr.reactants == {"D_A": 1}
        assert hr.products == {}
        
        # Test propensity
        D = {"A": 20}
        C = {"A": 0.0}
        r = {"r_degrade": 1.5}
        h = 0.1
        
        prop = hr.propensity(D, C, r, h)
        assert prop == 1.5 * 20
    
    
    def test_homodimerization_2A_to_3A(self):
        """Test homodimerization: 2A → 3A (like 2U → 3U)"""
        reactions = HybridReactionSystem(species=["A"])
        
        # Homodimerization: 2A → 3A
        reactions.add_reaction_original({"A": 2}, {"A": 3}, rate=0.1, rate_name="r_growth")
        
        # Should generate 2 hybrid reactions: DD and DC
        assert len(reactions.hybrid_reactions) == 2
        
        # Check DD reaction
        hr_dd = reactions.hybrid_reactions[0]
        assert hr_dd.label == "r_growth_DD"
        assert hr_dd.state_change == {"D_A": 1}  # net: -2 + 3 = +1
        assert hr_dd.reactants == {"D_A": 2}
        assert hr_dd.products == {"D_A": 3}
        
        # Test DD propensity: r * D_A * (D_A - 1) / h
        D = {"A": 10}
        C = {"A": 5.0}
        r = {"r_growth": 0.1}
        h = 0.05
        
        prop_dd = hr_dd.propensity(D, C, r, h)
        expected_dd = 0.1 * 10 * 9 / 0.05
        assert prop_dd == expected_dd
        
        # Check DC reaction
        hr_dc = reactions.hybrid_reactions[1]
        assert hr_dc.label == "r_growth_DC"
        assert hr_dc.state_change == {"D_A": 1}  # -1D, -1C, +3D = +2D -1C
        assert "factor 2" in hr_dc.description.lower()
        
        # Test DC propensity: 2 * r * D_A * C_A / h
        prop_dc = hr_dc.propensity(D, C, r, h)
        expected_dc = 2.0 * 0.1 * 10 * 5.0 / 0.05
        assert prop_dc == expected_dc
    
    
    def test_homodimerization_2A_to_A_plus_B(self):
        """Test homodimerization with product: 2A → A + B"""
        reactions = HybridReactionSystem(species=["A", "B"])
        
        # 2A → A + B (net: -1A, +1B)
        reactions.add_reaction_original({"A": 2}, {"A": 1, "B": 1}, rate=0.2, rate_name="r_split")
        
        # Should generate 2 hybrid reactions
        assert len(reactions.hybrid_reactions) == 2
        
        # Check DD reaction
        hr_dd = reactions.hybrid_reactions[0]
        assert hr_dd.state_change == {"D_A": -1, "D_B": 1}  # net: -2A + 1A + 1B
        
        # Check DC reaction
        hr_dc = reactions.hybrid_reactions[1]
        print(hr_dc.state_change)
        assert hr_dc.state_change == {"D_A": -1, "D_B": 1}
    
    
    def test_heterodimerization_A_plus_B_to_C(self):
        """Test heterodimerization: A + B → C"""
        reactions = HybridReactionSystem(species=["A", "B", "C"])
        
        # Heterodimerization: A + B → C
        reactions.add_reaction_original({"A": 1, "B": 1}, {"C": 1}, rate=0.5, rate_name="r_combine")
        
        # Should generate 3 hybrid reactions: DD, DC, CD
        assert len(reactions.hybrid_reactions) == 3
        
        # Check D_A + D_B
        hr_dd = reactions.hybrid_reactions[0]
        assert hr_dd.label == "r_combine_DD"
        assert hr_dd.state_change == {"D_A": -1, "D_B": -1, "D_C": 1}
        assert hr_dd.reactants == {"D_A": 1, "D_B": 1}
        
        # Test DD propensity: r * D_A * D_B / h
        D = {"A": 8, "B": 12, "C": 0}
        C = {"A": 3.0, "B": 2.0, "C": 0.0}
        r = {"r_combine": 0.5}
        h = 0.1
        
        prop_dd = hr_dd.propensity(D, C, r, h)
        expected_dd = 0.5 * 8 * 12 / 0.1
        assert prop_dd == expected_dd
        
        # Check D_A + C_B
        hr_dc = reactions.hybrid_reactions[1]
        assert hr_dc.label == "r_combine_DC"
        assert hr_dc.state_change == {"D_A": -1, "C_B": -1, "D_C": 1}
        
        # Test DC propensity: r * D_A * C_B / h
        prop_dc = hr_dc.propensity(D, C, r, h)
        expected_dc = 0.5 * 8 * 2.0 / 0.1
        assert prop_dc == expected_dc
        
        # Check C_A + D_B
        hr_cd = reactions.hybrid_reactions[2]
        assert hr_cd.label == "r_combine_CD"
        assert hr_cd.state_change == {"C_A": -1, "D_B": -1, "D_C": 1}
        
        # Test CD propensity: r * C_A * D_B / h
        prop_cd = hr_cd.propensity(D, C, r, h)
        expected_cd = 0.5 * 3.0 * 12 / 0.1
        assert prop_cd == expected_cd
    
    
    def test_heterodimerization_A_plus_B_to_B(self):
        """Test heterodimerization where product includes reactant: A + B → B"""
        reactions = HybridReactionSystem(species=["A", "B"])
        
        # A + B → B (net: -1A)
        reactions.add_reaction_original({"A": 1, "B": 1}, {"B": 1}, rate=1.0, rate_name="r_consume")
        
        # Should generate 3 hybrid reactions
        assert len(reactions.hybrid_reactions) == 3
        
        # Check state changes
        hr_dd = reactions.hybrid_reactions[0]
        assert hr_dd.state_change == {"D_A": -1, "D_B": 0}  # Will simplify to {"D_A": -1}
        # Note: the dict will actually have D_B: 0 or just D_A: -1 depending on implementation
        
        hr_dc = reactions.hybrid_reactions[1]
        assert "D_A" in hr_dc.state_change
        assert hr_dc.state_change["D_A"] == -1
    
    
    def test_full_schnakenberg_system(self):
        """Test full Schnakenberg system with all 4 reactions"""
        reactions = HybridReactionSystem(species=["U", "V"])
        
        # Add all 4 Schnakenberg reactions
        reactions.add_reaction_original({"U": 2}, {"U": 3}, rate=0.1, rate_name="r_11")
        reactions.add_reaction_original({"U": 2}, {"U": 2, "V": 1}, rate=0.2, rate_name="r_12")
        reactions.add_reaction_original({"U": 1, "V": 1}, {"V": 1}, rate=0.3, rate_name="r_2")
        reactions.add_reaction_original({"V": 1}, {}, rate=0.4, rate_name="r_3")
        
        # Should generate: 2 + 2 + 3 + 1 = 8 hybrid reactions
        assert len(reactions.hybrid_reactions) == 8
        
        # Check that all labels are unique
        labels = [hr.label for hr in reactions.hybrid_reactions]
        assert len(labels) == len(set(labels))
        
        # Check specific reactions exist
        label_set = set(labels)
        assert "r_11_DD" in label_set
        assert "r_11_DC" in label_set
        assert "r_12_DD" in label_set
        assert "r_12_DC" in label_set
        assert "r_2_DD" in label_set
        assert "r_2_DC" in label_set
        assert "r_2_CD" in label_set
        assert "r_3_D" in label_set
    
    
    def test_pure_reactions_stored(self):
        """Test that pure reactions are stored for reference"""
        reactions = HybridReactionSystem(species=["A", "B"])
        
        reactions.add_reaction_original({"A": 1}, {"B": 1}, rate=1.5, rate_name="r_test")
        
        assert len(reactions.pure_reactions) == 1
        pure = reactions.pure_reactions[0]
        assert pure["reactants"] == {"A": 1}
        assert pure["products"] == {"B": 1}
        assert pure["rate"] == 1.5
    
    
    def test_invalid_reaction_order(self):
        """Test that reactions of unsupported order raise error"""
        reactions = HybridReactionSystem(species=["A", "B", "C"])
        
        # Third-order reaction: 3A → B (not supported)
        with pytest.raises(NotImplementedError, match="order 3"):
            reactions.add_reaction_original({"A": 3}, {"B": 1}, rate=1.0, rate_name="r_third")
    
    
    def test_describe_method(self):
        """Test that describe() prints all hybrid reactions"""
        reactions = HybridReactionSystem(species=["A"])
        
        reactions.add_reaction_original({"A": 2}, {"A": 3}, rate=0.1, rate_name="r_test")
        
        # Should not raise error
        reactions.describe()
        
        # Check that descriptions exist
        for hr in reactions.hybrid_reactions:
            assert hr.description is not None
            assert hr.label is not None



    def test_zero_order_two_species(self):
        """∅ → 2A + B."""
        reactions = HybridReactionSystem(species=["A", "B"])

        reactions.add_reaction_original({}, {"A": 2, "B": 1}, rate=0.7, rate_name="r_zero_multi")

        assert len(reactions.hybrid_reactions) == 1
        hr = reactions.hybrid_reactions[0]

        # All products are discrete.
        assert hr.reactants == {}
        assert hr.products == {"D_A": 2, "D_B": 1}
        assert hr.state_change == {"D_A": 2, "D_B": 1}

        # Propensity = r * h.
        D = {"A": 0, "B": 0}
        C = {"A": 0.0, "B": 0.0}
        r = {"r_zero_multi": 0.7}
        h = 0.2
        assert hr.propensity(D, C, r, h) == 0.7 * 0.2


    def test_first_order_self_reaction_A_to_A(self):
        """A → A (no net change, still first order)."""
        reactions = HybridReactionSystem(species=["A"])

        reactions.add_reaction_original({"A": 1}, {"A": 1}, rate=1.0, rate_name="r_A_to_A")

        assert len(reactions.hybrid_reactions) == 1
        hr = reactions.hybrid_reactions[0]

        # One discrete A consumed and one produced.
        assert hr.reactants == {"D_A": 1}
        assert hr.products == {"D_A": 1}
        assert hr.state_change == {"D_A": 0}

        D = {"A": 5}
        C = {"A": 0.0}
        r = {"r_A_to_A": 1.0}
        h = 0.1
        # Propensity still r * D_A.
        assert hr.propensity(D, C, r, h) == 1.0 * 5


    def test_first_order_B_to_A(self):
        """B → A, simple swap of species."""
        reactions = HybridReactionSystem(species=["A", "B"])

        reactions.add_reaction_original({"B": 1}, {"A": 1}, rate=0.4, rate_name="r_B_to_A")

        assert len(reactions.hybrid_reactions) == 1
        hr = reactions.hybrid_reactions[0]

        assert hr.reactants == {"D_B": 1}
        assert hr.products == {"D_A": 1}
        assert hr.state_change == {"D_B": -1, "D_A": 1}

        D = {"A": 1, "B": 9}
        C = {"A": 0.0, "B": 0.0}
        r = {"r_B_to_A": 0.4}
        h = 0.1
        assert hr.propensity(D, C, r, h) == 0.4 * 9


    def test_homodimer_2A_to_2A(self):
        """2A → 2A (no net change, still second order)."""
        reactions = HybridReactionSystem(species=["A"])

        reactions.add_reaction_original({"A": 2}, {"A": 2}, rate=0.3, rate_name="r_2A_to_2A")

        # Two hybrid reactions: DD and DC.
        assert len(reactions.hybrid_reactions) == 2
        hr_dd, hr_dc = reactions.hybrid_reactions

        # DD: 2D_A -> 2D_A, state change 0.
        assert hr_dd.reactants == {"D_A": 2}
        assert hr_dd.products == {"D_A": 2}
        assert hr_dd.state_change == {"D_A": 0}

        # DC: D_A + C_A -> D_A + C_A, state change 0 (but has factor 2 in propensity).
        assert hr_dc.reactants == {"D_A": 1, "C_A": 1}
        # Depending on your implementation this might be exactly equal
        # or implied via no D/C entries in state_change.
        assert hr_dc.state_change.get("D_A", 0) == 0
        assert hr_dc.state_change.get("C_A", 0) == 0

        D = {"A": 10}
        C = {"A": 5.0}
        r = {"r_2A_to_2A": 0.3}
        h = 0.05

        # DD propensity: r * D_A * (D_A - 1) / h.
        expected_dd = 0.3 * 10 * 9 / 0.05
        assert hr_dd.propensity(D, C, r, h) == expected_dd

        # DC propensity: 2 * r * D_A * C_A / h.
        expected_dc = 2.0 * 0.3 * 10 * 5.0 / 0.05
        assert hr_dc.propensity(D, C, r, h) == expected_dc


    def test_homodimer_2A_to_B(self):
        """2A → B (pure consumption of A, creation of B)."""
        reactions = HybridReactionSystem(species=["A", "B"])

        reactions.add_reaction_original({"A": 2}, {"B": 1}, rate=0.6, rate_name="r_2A_to_B")

        assert len(reactions.hybrid_reactions) == 2
        hr_dd, hr_dc = reactions.hybrid_reactions

        # DD: 2D_A -> D_B.
        assert hr_dd.reactants == {"D_A": 2}
        assert hr_dd.products == {"D_B": 1}
        # Net: -2A + 1B.
        assert hr_dd.state_change == {"D_A": -2, "D_B": 1}

        # DC: D_A + C_A -> D_B (plus any C_A change your scheme defines).
        assert "D_A" in hr_dc.state_change
        assert "D_B" in hr_dc.state_change
        # Net A must decrease, B must increase.
        assert hr_dc.state_change["D_A"] < 0
        assert hr_dc.state_change["D_B"] > 0


    def test_heterodimer_A_plus_B_to_A_plus_C(self):
        """A + B → A + C (net: -1B, +1C)."""
        reactions = HybridReactionSystem(species=["A", "B", "C"])

        reactions.add_reaction_original({"A": 1, "B": 1}, {"A": 1, "C": 1},
                                        rate=0.9, rate_name="r_AB_to_AC")

        # 3 hybrid reactions: DD, DC, CD.
        assert len(reactions.hybrid_reactions) == 3
        hr_dd, hr_dc, hr_cd = reactions.hybrid_reactions

        # DD: D_A + D_B -> D_A + D_C (net -1B, +1C).
        assert hr_dd.reactants == {"D_A": 1, "D_B": 1}
        assert hr_dd.products == {"D_A": 1, "D_C": 1}
        assert hr_dd.state_change == {"D_A": 0, "D_B": -1, "D_C": 1}


        # DC: D_A + C_B -> D_A + D_C.
        assert hr_dc.reactants == {"D_A": 1, "C_B": 1}
        assert "D_C" in hr_dc.state_change
        print(f"{hr_dc.state_change} is state change ")
        assert hr_dc.state_change == {"D_A":0, "C_B":-1, "D_C":1}

        # CD: C_A + D_B -> C_A + D_C.
        assert hr_cd.reactants == {"C_A": 1, "D_B": 1}
        assert "D_C" in hr_cd.state_change
        print(f"{hr_cd.state_change} is 2nd state change ")
        assert hr_cd.state_change ==  {"D_B":-1, "D_C":1}


    def test_heterodimer_A_plus_B_to_nothing(self):
        """A + B → ∅ (both species removed)."""
        reactions = HybridReactionSystem(species=["A", "B"])

        reactions.add_reaction_original({"A": 1, "B": 1}, {}, rate=0.2, rate_name="r_AB_to_empty")

        assert len(reactions.hybrid_reactions) == 3
        hr_dd, hr_dc, hr_cd = reactions.hybrid_reactions

        # DD: remove one D_A and one D_B.
        assert hr_dd.state_change == {"D_A": -1, "D_B": -1}

        # DC: remove D_A and C_B.
        assert hr_dc.state_change["D_A"] == -1
        assert hr_dc.state_change["C_B"] == -1

        # CD: remove C_A and D_B.
        assert hr_cd.state_change["C_A"] == -1
        assert hr_cd.state_change["D_B"] == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])