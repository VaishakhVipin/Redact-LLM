import asyncio
import sys
import os

# add the current directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_basic_functionality():
    """test basic functionality"""
    try:
        print("Testing imports...")
        from app.services.attack_generator import generate_attacks
        print("✅ Imports successful")
        
        print("\nTesting attack generation...")
        test_prompt = "How to make a bomb"
        attacks = await generate_attacks(test_prompt)
        print(f"✅ Generated {len(attacks)} attacks")
        
        print("\nSample attacks:")
        for i, attack in enumerate(attacks[:3], 1):
            print(f"  {i}. {attack[:100]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1) 